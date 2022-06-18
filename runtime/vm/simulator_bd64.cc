// Copyright (c) 2014, the Dart project authors.  Please see the AUTHORS file
// for details. All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <setjmp.h>  // NOLINT
#include <stdlib.h>

#include "vm/globals.h"
#if defined(TARGET_ARCH_BD64)

// Only build the simulator if not compiling for real ARM hardware.
#if defined(USING_SIMULATOR)

#include "vm/simulator.h"

#include "vm/compiler/assembler/disassembler.h"
#include "vm/constants.h"
#include "vm/image_snapshot.h"
#include "vm/native_arguments.h"
#include "vm/os_thread.h"
#include "vm/stack_frame.h"

namespace dart {

// constants_arm64.h does not define LR constant to prevent accidental direct
// use of it during code generation. However using LR directly is okay in this
// file because it is a simulator.
constexpr Register LR = LR_DO_NOT_USE_DIRECTLY;

// This macro provides a platform independent use of sscanf. The reason for
// SScanF not being implemented in a platform independent way through
// OS in the same way as SNPrint is that the Windows C Run-Time
// Library does not provide vsscanf.
#define SScanF sscanf  // NOLINT

// SimulatorSetjmpBuffer are linked together, and the last created one
// is referenced by the Simulator. When an exception is thrown, the exception
// runtime looks at where to jump and finds the corresponding
// SimulatorSetjmpBuffer based on the stack pointer of the exception handler.
// The runtime then does a Longjmp on that buffer to return to the simulator.
class SimulatorSetjmpBuffer {
 public:
  void Longjmp() {
    // "This" is now the last setjmp buffer.
    simulator_->set_last_setjmp_buffer(this);
    longjmp(buffer_, 1);
  }

  explicit SimulatorSetjmpBuffer(Simulator* sim) {
    simulator_ = sim;
    link_ = sim->last_setjmp_buffer();
    sim->set_last_setjmp_buffer(this);
    sp_ = static_cast<uword>(sim->get_register(R31));
  }

  ~SimulatorSetjmpBuffer() {
    ASSERT(simulator_->last_setjmp_buffer() == this);
    simulator_->set_last_setjmp_buffer(link_);
  }

  SimulatorSetjmpBuffer* link() { return link_; }

  uword sp() { return sp_; }

 private:
  uword sp_;
  Simulator* simulator_;
  SimulatorSetjmpBuffer* link_;
  jmp_buf buffer_;

  friend class Simulator;
};

void Simulator::Init() {}

Simulator::Simulator() : exclusive_access_addr_(0), exclusive_access_value_(0) {
  // Setup simulator support first. Some of this information is needed to
  // setup the architecture state.
  // We allocate the stack here, the size is computed as the sum of
  // the size specified by the user and the buffer space needed for
  // handling stack overflow exceptions. To be safe in potential
  // stack underflows we also add some underflow buffer space.
  stack_ =
      new char[(OSThread::GetSpecifiedStackSize() +
                OSThread::kStackSizeBufferMax + kSimulatorStackUnderflowSize)];
  // Low address.
  stack_limit_ = reinterpret_cast<uword>(stack_);
  // Limit for StackOverflowError.
  overflow_stack_limit_ = stack_limit_ + OSThread::kStackSizeBufferMax;
  // High address.
  stack_base_ = overflow_stack_limit_ + OSThread::GetSpecifiedStackSize();

  pc_modified_ = false;
  icount_ = 0;
  break_pc_ = NULL;
  break_instr_ = 0;
  last_setjmp_buffer_ = NULL;

  // Setup architecture state.
  // All registers are initialized to zero to start with.
  for (int i = 0; i < kNumberOfCpuRegisters; i++) {
    registers_[i] = 0;
  }
  n_flag_ = false;
  z_flag_ = false;
  c_flag_ = false;
  v_flag_ = false;

  for (int i = 0; i < kNumberOfVRegisters; i++) {
    vregisters_[i].bits.i64[0] = 0;
    vregisters_[i].bits.i64[1] = 0;
  }

  // The sp is initialized to point to the bottom (high address) of the
  // allocated stack area.
  registers_[R31] = stack_base();
  // The lr and pc are initialized to a known bad value that will cause an
  // access violation if the simulator ever tries to execute it.
  registers_[LR] = kBadLR;
  pc_ = kBadLR;
}

Simulator::~Simulator() {
  delete[] stack_;
  Isolate* isolate = Isolate::Current();
  if (isolate != NULL) {
    isolate->set_simulator(NULL);
  }
}

// When the generated code calls an external reference we need to catch that in
// the simulator.  The external reference will be a function compiled for the
// host architecture.  We need to call that function instead of trying to
// execute it with the simulator.  We do that by redirecting the external
// reference to a svc (supervisor call) instruction that is handled by
// the simulator.  We write the original destination of the jump just at a known
// offset from the svc instruction so the simulator knows what to call.
class Redirection {
 public:
  uword address_of_hlt_instruction() {
    return reinterpret_cast<uword>(&hlt_instruction_);
  }

  uword external_function() const { return external_function_; }

  Simulator::CallKind call_kind() const { return call_kind_; }

  int argument_count() const { return argument_count_; }

  static Redirection* Get(uword external_function,
                          Simulator::CallKind call_kind,
                          int argument_count) {
    MutexLocker ml(mutex_);

    Redirection* old_head = list_.load(std::memory_order_relaxed);
    for (Redirection* current = old_head; current != nullptr;
         current = current->next_) {
      if (current->external_function_ == external_function) return current;
    }

    Redirection* redirection =
        new Redirection(external_function, call_kind, argument_count);
    redirection->next_ = old_head;

    // Use a memory fence to ensure all pending writes are written at the time
    // of updating the list head, so the profiling thread always has a valid
    // list to look at.
    list_.store(redirection, std::memory_order_release);

    return redirection;
  }

  static Redirection* FromHltInstruction(Instr* hlt_instruction) {
    char* addr_of_hlt = reinterpret_cast<char*>(hlt_instruction);
    char* addr_of_redirection =
        addr_of_hlt - OFFSET_OF(Redirection, hlt_instruction_);
    return reinterpret_cast<Redirection*>(addr_of_redirection);
  }

  // Please note that this function is called by the signal handler of the
  // profiling thread.  It can therefore run at any point in time and is not
  // allowed to hold any locks - which is precisely the reason why the list is
  // prepend-only and a memory fence is used when writing the list head [list_]!
  static uword FunctionForRedirect(uword address_of_hlt) {
    for (Redirection* current = list_.load(std::memory_order_acquire);
         current != nullptr; current = current->next_) {
      if (current->address_of_hlt_instruction() == address_of_hlt) {
        return current->external_function_;
      }
    }
    return 0;
  }

 private:
  Redirection(uword external_function,
              Simulator::CallKind call_kind,
              int argument_count)
      : external_function_(external_function),
        call_kind_(call_kind),
        argument_count_(argument_count),
        hlt_instruction_(Instr::kSimulatorRedirectInstruction),
        next_(NULL) {}

  uword external_function_;
  Simulator::CallKind call_kind_;
  int argument_count_;
  uint32_t hlt_instruction_;
  Redirection* next_;
  static std::atomic<Redirection*> list_;
  static Mutex* mutex_;
};

std::atomic<Redirection*> Redirection::list_ = {nullptr};
Mutex* Redirection::mutex_ = new Mutex();

uword Simulator::RedirectExternalReference(uword function,
                                           CallKind call_kind,
                                           int argument_count) {
  Redirection* redirection =
      Redirection::Get(function, call_kind, argument_count);
  return redirection->address_of_hlt_instruction();
}

uword Simulator::FunctionForRedirect(uword redirect) {
  return Redirection::FunctionForRedirect(redirect);
}

// Get the active Simulator for the current isolate.
Simulator* Simulator::Current() {
  Isolate* isolate = Isolate::Current();
  Simulator* simulator = isolate->simulator();
  if (simulator == NULL) {
    NoSafepointScope no_safepoint;
    simulator = new Simulator();
    isolate->set_simulator(simulator);
  }
  return simulator;
}

// Sets the register in the architecture state.
DART_FORCE_INLINE
void Simulator::set_register(Register reg, int64_t value) {
  ASSERT((reg >= 0) && (reg < kNumberOfCpuRegisters));
  if (LIKELY(reg != ZR)) {
    registers_[reg] = value;
  }
}

DART_FORCE_INLINE
void Simulator::set_wregister(Register reg, int32_t value) {
  ASSERT((reg >= 0) && (reg < kNumberOfCpuRegisters));
  // When setting in W mode, clear the high bits.
  if (LIKELY(reg != ZR)) {
    registers_[reg] = Utils::LowHighTo64Bits(static_cast<uint32_t>(value), 0);
  }
}

// Get the register from the architecture state.
DART_FORCE_INLINE
int32_t Simulator::get_wregister(Register reg) const {
  ASSERT((reg >= 0) && (reg < kNumberOfCpuRegisters));
  return static_cast<int32_t>(registers_[reg]);
}

DART_FORCE_INLINE
int32_t Simulator::get_vregisters(VRegister reg, int idx) const {
  ASSERT((reg >= 0) && (reg < kNumberOfVRegisters));
  ASSERT((idx >= 0) && (idx <= 3));
  return vregisters_[reg].bits.i32[idx];
}

DART_FORCE_INLINE
void Simulator::set_vregisters(VRegister reg, int idx, int32_t value) {
  ASSERT((reg >= 0) && (reg < kNumberOfVRegisters));
  ASSERT((idx >= 0) && (idx <= 3));
  vregisters_[reg].bits.i32[idx] = value;
}

DART_FORCE_INLINE
int64_t Simulator::get_vregisterd(VRegister reg, int idx) const {
  ASSERT((reg >= 0) && (reg < kNumberOfVRegisters));
  ASSERT((idx == 0) || (idx == 1));
  return vregisters_[reg].bits.i64[idx];
}

DART_FORCE_INLINE
void Simulator::set_vregisterd(VRegister reg, int idx, int64_t value) {
  ASSERT((reg >= 0) && (reg < kNumberOfVRegisters));
  ASSERT((idx == 0) || (idx == 1));
  vregisters_[reg].bits.i64[idx] = value;
}

DART_FORCE_INLINE
void Simulator::get_vregister(VRegister reg, simd_value_t* value) const {
  ASSERT((reg >= 0) && (reg < kNumberOfVRegisters));
  value->bits.i64[0] = vregisters_[reg].bits.i64[0];
  value->bits.i64[1] = vregisters_[reg].bits.i64[1];
}

DART_FORCE_INLINE
void Simulator::set_vregister(VRegister reg, const simd_value_t& value) {
  ASSERT((reg >= 0) && (reg < kNumberOfVRegisters));
  vregisters_[reg].bits.i64[0] = value.bits.i64[0];
  vregisters_[reg].bits.i64[1] = value.bits.i64[1];
}

// Raw access to the PC register.
DART_FORCE_INLINE
void Simulator::set_pc(uint64_t value) {
  pc_modified_ = true;
  last_pc_ = pc_;
  pc_ = value;
}

void Simulator::UnimplementedInstruction(Instr* instr) {
  OS::PrintErr("Unimplemented instruction: at %p, last_pc=0x%" Px64 "\n", instr,
               get_last_pc());
  FATAL("Cannot continue execution after unimplemented instruction.");
}

DART_FORCE_INLINE
intptr_t Simulator::ReadX(uword addr,
                          Instr* instr,
                          bool must_be_aligned /* = false */) {
  intptr_t* ptr = reinterpret_cast<intptr_t*>(addr);
  return *ptr;
}

DART_FORCE_INLINE
void Simulator::WriteX(uword addr, intptr_t value, Instr* instr) {
  intptr_t* ptr = reinterpret_cast<intptr_t*>(addr);
  *ptr = value;
}

DART_FORCE_INLINE
uint32_t Simulator::ReadWU(uword addr,
                           Instr* instr,
                           bool must_be_aligned /* = false */) {
  uint32_t* ptr = reinterpret_cast<uint32_t*>(addr);
  return *ptr;
}

DART_FORCE_INLINE
int32_t Simulator::ReadW(uword addr, Instr* instr) {
  int32_t* ptr = reinterpret_cast<int32_t*>(addr);
  return *ptr;
}

DART_FORCE_INLINE
void Simulator::WriteW(uword addr, uint32_t value, Instr* instr) {
  uint32_t* ptr = reinterpret_cast<uint32_t*>(addr);
  *ptr = value;
}

DART_FORCE_INLINE
uint16_t Simulator::ReadHU(uword addr, Instr* instr) {
  uint16_t* ptr = reinterpret_cast<uint16_t*>(addr);
  return *ptr;
}

DART_FORCE_INLINE
int16_t Simulator::ReadH(uword addr, Instr* instr) {
  int16_t* ptr = reinterpret_cast<int16_t*>(addr);
  return *ptr;
}

DART_FORCE_INLINE
void Simulator::WriteH(uword addr, uint16_t value, Instr* instr) {
  uint16_t* ptr = reinterpret_cast<uint16_t*>(addr);
  *ptr = value;
}

DART_FORCE_INLINE
uint8_t Simulator::ReadBU(uword addr) {
  uint8_t* ptr = reinterpret_cast<uint8_t*>(addr);
  return *ptr;
}

DART_FORCE_INLINE
int8_t Simulator::ReadB(uword addr) {
  int8_t* ptr = reinterpret_cast<int8_t*>(addr);
  return *ptr;
}

DART_FORCE_INLINE
void Simulator::WriteB(uword addr, uint8_t value) {
  uint8_t* ptr = reinterpret_cast<uint8_t*>(addr);
  *ptr = value;
}

DART_FORCE_INLINE
void Simulator::ClearExclusive() {
  exclusive_access_addr_ = 0;
  exclusive_access_value_ = 0;
}

DART_FORCE_INLINE
intptr_t Simulator::ReadExclusiveX(uword addr, Instr* instr) {
  exclusive_access_addr_ = addr;
  exclusive_access_value_ = ReadX(addr, instr, /*must_be_aligned=*/true);
  return exclusive_access_value_;
}

DART_FORCE_INLINE
intptr_t Simulator::ReadExclusiveW(uword addr, Instr* instr) {
  exclusive_access_addr_ = addr;
  exclusive_access_value_ = ReadWU(addr, instr, /*must_be_aligned=*/true);
  return exclusive_access_value_;
}

DART_FORCE_INLINE
intptr_t Simulator::WriteExclusiveX(uword addr, intptr_t value, Instr* instr) {
  // In a well-formed code store-exclusive instruction should always follow
  // a corresponding load-exclusive instruction with the same address.
  ASSERT((exclusive_access_addr_ == 0) || (exclusive_access_addr_ == addr));
  if (exclusive_access_addr_ != addr) {
    return 1;  // Failure.
  }

  int64_t old_value = exclusive_access_value_;
  ClearExclusive();

  auto atomic_addr = reinterpret_cast<RelaxedAtomic<int64_t>*>(addr);
  if (atomic_addr->compare_exchange_weak(old_value, value)) {
    return 0;  // Success.
  }
  return 1;  // Failure.
}

DART_FORCE_INLINE
intptr_t Simulator::WriteExclusiveW(uword addr, intptr_t value, Instr* instr) {
  // In a well-formed code store-exclusive instruction should always follow
  // a corresponding load-exclusive instruction with the same address.
  ASSERT((exclusive_access_addr_ == 0) || (exclusive_access_addr_ == addr));
  if (exclusive_access_addr_ != addr) {
    return 1;  // Failure.
  }

  int32_t old_value = static_cast<uint32_t>(exclusive_access_value_);
  ClearExclusive();

  auto atomic_addr = reinterpret_cast<RelaxedAtomic<int32_t>*>(addr);
  if (atomic_addr->compare_exchange_weak(old_value, value)) {
    return 0;  // Success.
  }
  return 1;  // Failure.
}

DART_FORCE_INLINE
intptr_t Simulator::ReadAcquire(uword addr, Instr* instr) {
  // TODO(42074): Once we switch to C++20 we should change this to use use
  // `std::atomic_ref<T>` which supports performing atomic operations on
  // non-atomic data.
  COMPILE_ASSERT(sizeof(std::atomic<intptr_t>) == sizeof(intptr_t));
  return reinterpret_cast<std::atomic<intptr_t>*>(addr)->load(
      std::memory_order_acquire);
}

DART_FORCE_INLINE
uint32_t Simulator::ReadAcquireW(uword addr, Instr* instr) {
  // TODO(42074): Once we switch to C++20 we should change this to use use
  // `std::atomic_ref<T>` which supports performing atomic operations on
  // non-atomic data.
  COMPILE_ASSERT(sizeof(std::atomic<intptr_t>) == sizeof(intptr_t));
  return reinterpret_cast<std::atomic<uint32_t>*>(addr)->load(
      std::memory_order_acquire);
}

DART_FORCE_INLINE
void Simulator::WriteRelease(uword addr, intptr_t value, Instr* instr) {
  // TODO(42074): Once we switch to C++20 we should change this to use use
  // `std::atomic_ref<T>` which supports performing atomic operations on
  // non-atomic data.
  COMPILE_ASSERT(sizeof(std::atomic<intptr_t>) == sizeof(intptr_t));
  reinterpret_cast<std::atomic<intptr_t>*>(addr)->store(
      value, std::memory_order_release);
}

DART_FORCE_INLINE
void Simulator::WriteReleaseW(uword addr, uint32_t value, Instr* instr) {
  // TODO(42074): Once we switch to C++20 we should change this to use use
  // `std::atomic_ref<T>` which supports performing atomic operations on
  // non-atomic data.
  COMPILE_ASSERT(sizeof(std::atomic<intptr_t>) == sizeof(intptr_t));
  reinterpret_cast<std::atomic<uint32_t>*>(addr)->store(
      value, std::memory_order_release);
}

// Unsupported instructions use Format to print an error and stop execution.
void Simulator::Format(Instr* instr, const char* format) {
  OS::PrintErr("Simulator found unsupported instruction:\n 0x%p: %s\n", instr,
               format);
  UNIMPLEMENTED();
}

// Calculate and set the Negative and Zero flags.
DART_FORCE_INLINE
void Simulator::SetNZFlagsW(int32_t val) {
  n_flag_ = (val < 0);
  z_flag_ = (val == 0);
}

// Calculate C flag value for additions (and subtractions with adjusted args).
DART_FORCE_INLINE
bool Simulator::CarryFromW(int32_t left, int32_t right, int32_t carry) {
  uint64_t uleft = static_cast<uint32_t>(left);
  uint64_t uright = static_cast<uint32_t>(right);
  uint64_t ucarry = static_cast<uint32_t>(carry);
  return ((uleft + uright + ucarry) >> 32) != 0;
}

// Calculate V flag value for additions (and subtractions with adjusted args).
DART_FORCE_INLINE
bool Simulator::OverflowFromW(int32_t left, int32_t right, int32_t carry) {
  int64_t result = static_cast<int64_t>(left) + right + carry;
  return (result >> 31) != (result >> 32);
}

// Calculate and set the Negative and Zero flags.
DART_FORCE_INLINE
void Simulator::SetNZFlagsX(int64_t val) {
  n_flag_ = (val < 0);
  z_flag_ = (val == 0);
}

// Calculate C flag value for additions and subtractions.
DART_FORCE_INLINE
bool Simulator::CarryFromX(int64_t alu_out,
                           int64_t left,
                           int64_t right,
                           bool addition) {
  if (addition) {
    return (((left & right) | ((left | right) & ~alu_out)) >> 63) != 0;
  } else {
    return (((~left & right) | ((~left | right) & alu_out)) >> 63) == 0;
  }
}

// Calculate V flag value for additions and subtractions.
DART_FORCE_INLINE
bool Simulator::OverflowFromX(int64_t alu_out,
                              int64_t left,
                              int64_t right,
                              bool addition) {
  if (addition) {
    return (((alu_out ^ left) & (alu_out ^ right)) >> 63) != 0;
  } else {
    return (((left ^ right) & (alu_out ^ left)) >> 63) != 0;
  }
}

// Set the Carry flag.
DART_FORCE_INLINE
void Simulator::SetCFlag(bool val) {
  c_flag_ = val;
}

// Set the oVerflow flag.
DART_FORCE_INLINE
void Simulator::SetVFlag(bool val) {
  v_flag_ = val;
}

void Simulator::DecodeMoveWide(Instr* instr) {
  const Register rd = instr->RdField();
  const int hw = instr->HWField();
  const int64_t shift = hw << 4;
  const int64_t shifted_imm = static_cast<int64_t>(instr->Imm16Field())
                              << shift;

  if (instr->SFField()) {
    if (instr->Bits(29, 2) == 0) {
      // Format(instr, "movn'sf 'rd, 'imm16 'hw");
      set_register(rd, ~shifted_imm);
    } else if (instr->Bits(29, 2) == 2) {
      // Format(instr, "movz'sf 'rd, 'imm16 'hw");
      set_register(rd, shifted_imm);
    } else if (instr->Bits(29, 2) == 3) {
      // Format(instr, "movk'sf 'rd, 'imm16 'hw");
      const int64_t rd_val = get_register(rd);
      const int64_t result = (rd_val & ~(0xffffL << shift)) | shifted_imm;
      set_register(rd, result);
    }
  } else if ((hw & 0x2) == 0) {
    if (instr->Bits(29, 2) == 0) {
      // Format(instr, "movn'sf 'rd, 'imm16 'hw");
      set_wregister(rd, ~shifted_imm & kWRegMask);
    } else if (instr->Bits(29, 2) == 2) {
      // Format(instr, "movz'sf 'rd, 'imm16 'hw");
      set_wregister(rd, shifted_imm & kWRegMask);
    } else if (instr->Bits(29, 2) == 3) {
      // Format(instr, "movk'sf 'rd, 'imm16 'hw");
      const int32_t rd_val = get_wregister(rd);
      const int32_t result = (rd_val & ~(0xffffL << shift)) | shifted_imm;
      set_wregister(rd, result);
    }
  }
}

void Simulator::DecodeAddSubImm(Instr* instr) {
  const bool addition = (instr->Bit(30) == 0);
  // Format(instr, "addi'sf's 'rd, 'rn, 'imm12s");
  // Format(instr, "subi'sf's 'rd, 'rn, 'imm12s");
  const Register rd = instr->RdField();
  const Register rn = instr->RnField();
  uint32_t imm = (instr->Bit(22) == 1) ? (instr->Imm12Field() << 12)
                                       : (instr->Imm12Field());
  if (instr->SFField()) {
    // 64-bit add.
    const uint64_t rn_val = get_register(rn);
    const uint64_t alu_out = addition ? (rn_val + imm) : (rn_val - imm);
    set_register(rd, alu_out);
    if (instr->HasS()) {
      SetNZFlagsX(alu_out);
      SetCFlag(CarryFromX(alu_out, rn_val, imm, addition));
      SetVFlag(OverflowFromX(alu_out, rn_val, imm, addition));
    }
  } else {
    // 32-bit add.
    const uint32_t rn_val = get_wregister(rn);
    uint32_t carry_in = 0;
    if (!addition) {
      carry_in = 1;
      imm = ~imm;
    }
    const uint32_t alu_out = rn_val + imm + carry_in;
    set_wregister(rd, alu_out);
    if (instr->HasS()) {
      SetNZFlagsW(alu_out);
      SetCFlag(CarryFromW(rn_val, imm, carry_in));
      SetVFlag(OverflowFromW(rn_val, imm, carry_in));
    }
  }
}

void Simulator::DecodeBitfield(Instr* instr) {
  int bitwidth = instr->SFField() == 0 ? 32 : 64;
  unsigned op = instr->Bits(29, 2);
  ASSERT(op <= 2);
  bool sign_extend = op == 0;
  bool zero_extend = op == 2;
  ASSERT(instr->NField() == instr->SFField());
  const Register rn = instr->RnField();
  const Register rd = instr->RdField();
  int64_t result = get_register(rn);
  int r_bit = instr->ImmRField();
  int s_bit = instr->ImmSField();
  result &= Utils::NBitMask(bitwidth);
  ASSERT(s_bit < bitwidth && r_bit < bitwidth);
  // See ARM v8 Instruction set overview 5.4.5.
  // If s >= r then Rd[s-r:0] := Rn[s:r], else Rd[bitwidth+s-r:bitwidth-r] :=
  // Rn[s:0].
  uword mask = Utils::NBitMask(s_bit + 1);
  if (s_bit >= r_bit) {
    mask >>= r_bit;
    result >>= r_bit;
  } else {
    result = static_cast<uint64_t>(result) << (bitwidth - r_bit);
    mask <<= bitwidth - r_bit;
  }
  result &= mask;
  if (sign_extend) {
    int highest_bit = (s_bit - r_bit) & (bitwidth - 1);
    int shift = 64 - highest_bit - 1;
    result <<= shift;
    result = static_cast<word>(result) >> shift;
  } else if (!zero_extend) {
    const int64_t rd_val = get_register(rd);
    result |= rd_val & ~mask;
  }
  if (bitwidth == 64) {
    set_register(rd, result);
  } else {
    set_wregister(rd, result);
  }
}

void Simulator::DecodeLogicalImm(Instr* instr) {
  const int op = instr->Bits(29, 2);
  const bool set_flags = op == 3;
  const int out_size = ((instr->SFField() == 0) && (instr->NField() == 0))
                           ? kWRegSizeInBits
                           : kXRegSizeInBits;
  const Register rn = instr->RnField();
  const Register rd = instr->RdField();
  const int64_t rn_val = get_register(rn);
  const uint64_t imm = instr->ImmLogical();

  int64_t alu_out = 0;
  switch (op) {
    case 0:
      alu_out = rn_val & imm;
      break;
    case 1:
      alu_out = rn_val | imm;
      break;
    case 2:
      alu_out = rn_val ^ imm;
      break;
    case 3:
      alu_out = rn_val & imm;
      break;
    default:
      UNREACHABLE();
      break;
  }

  if (set_flags) {
    if (out_size == kXRegSizeInBits) {
      SetNZFlagsX(alu_out);
    } else {
      SetNZFlagsW(alu_out);
    }
    SetCFlag(false);
    SetVFlag(false);
  }

  if (out_size == kXRegSizeInBits) {
    set_register(rd, alu_out);
  } else {
    set_wregister(rd, alu_out);
  }
}

void Simulator::DecodePCRel(Instr* instr) {
  const int op = instr->Bit(31);
  if (LIKELY(op == 0)) {
    // Format(instr, "adr 'rd, 'pcrel")
    const Register rd = instr->RdField();
    const uint64_t immhi = instr->SImm19Field();
    const uint64_t immlo = instr->Bits(29, 2);
    const uint64_t off = (immhi << 2) | immlo;
    const uint64_t dest = get_pc() + off;
    set_register(rd, dest);
  }
}

void Simulator::DecodeDPImmediate(Instr* instr) {
  if (instr->IsMoveWideOp()) {
    DecodeMoveWide(instr);
  } else if (instr->IsAddSubImmOp()) {
    DecodeAddSubImm(instr);
  } else if (instr->IsBitfieldOp()) {
    DecodeBitfield(instr);
  } else if (instr->IsLogicalImmOp()) {
    DecodeLogicalImm(instr);
  } else if (instr->IsPCRelOp()) {
    DecodePCRel(instr);
  }
}

void Simulator::DecodeCompareAndBranch(Instr* instr) {
  const int op = instr->Bit(24);
  const Register rt = instr->RtField();
  const uint64_t imm19 = instr->SImm19Field();
  const uint64_t dest = get_pc() + (imm19 << 2);
  const uint64_t mask = instr->SFField() == 1 ? kXRegMask : kWRegMask;
  const uint64_t rt_val = get_register(rt) & mask;
  if (op == 0) {
    // Format(instr, "cbz'sf 'rt, 'dest19");
    if (rt_val == 0) {
      set_pc(dest);
    }
  } else {
    // Format(instr, "cbnz'sf 'rt, 'dest19");
    if (rt_val != 0) {
      set_pc(dest);
    }
  }
}

bool Simulator::ConditionallyExecute(Instr* instr) {
  Condition cond;
  if (instr->IsConditionalSelectOp()) {
    cond = instr->SelectConditionField();
  } else {
    cond = instr->ConditionField();
  }
  switch (cond) {
    case EQ:
      return z_flag_;
    case NE:
      return !z_flag_;
    case CS:
      return c_flag_;
    case CC:
      return !c_flag_;
    case MI:
      return n_flag_;
    case PL:
      return !n_flag_;
    case VS:
      return v_flag_;
    case VC:
      return !v_flag_;
    case HI:
      return c_flag_ && !z_flag_;
    case LS:
      return !c_flag_ || z_flag_;
    case GE:
      return n_flag_ == v_flag_;
    case LT:
      return n_flag_ != v_flag_;
    case GT:
      return !z_flag_ && (n_flag_ == v_flag_);
    case LE:
      return z_flag_ || (n_flag_ != v_flag_);
    case AL:
      return true;
    default:
      UNREACHABLE();
  }
  return false;
}

void Simulator::DecodeConditionalBranch(Instr* instr) {
  // Format(instr, "b'cond 'dest19");
  const uint64_t imm19 = instr->SImm19Field();
  const uint64_t dest = get_pc() + (imm19 << 2);
  if (ConditionallyExecute(instr)) {
    set_pc(dest);
  }
}

// Calls into the Dart runtime are based on this interface.
typedef void (*SimulatorRuntimeCall)(NativeArguments arguments);

// Calls to leaf Dart runtime functions are based on this interface.
typedef int64_t (*SimulatorLeafRuntimeCall)(int64_t r0,
                                            int64_t r1,
                                            int64_t r2,
                                            int64_t r3,
                                            int64_t r4,
                                            int64_t r5,
                                            int64_t r6,
                                            int64_t r7);

// [target] has several different signatures that differ from
// SimulatorLeafRuntimeCall. We can call them all from here only because in
// X64's calling conventions a function can be called with extra arguments
// and the callee will see the first arguments and won't unbalance the stack.
NO_SANITIZE_UNDEFINED("function")
static int64_t InvokeLeafRuntime(SimulatorLeafRuntimeCall target,
                                 int64_t r0,
                                 int64_t r1,
                                 int64_t r2,
                                 int64_t r3,
                                 int64_t r4,
                                 int64_t r5,
                                 int64_t r6,
                                 int64_t r7) {
  return target(r0, r1, r2, r3, r4, r5, r6, r7);
}

// Calls to leaf float Dart runtime functions are based on this interface.
typedef double (*SimulatorLeafFloatRuntimeCall)(double d0,
                                                double d1,
                                                double d2,
                                                double d3,
                                                double d4,
                                                double d5,
                                                double d6,
                                                double d7);

// [target] has several different signatures that differ from
// SimulatorFloatLeafRuntimeCall. We can call them all from here only because in
// X64's calling conventions a function can be called with extra arguments
// and the callee will see the first arguments and won't unbalance the stack.
NO_SANITIZE_UNDEFINED("function")
static double InvokeFloatLeafRuntime(SimulatorLeafFloatRuntimeCall target,
                                     double d0,
                                     double d1,
                                     double d2,
                                     double d3,
                                     double d4,
                                     double d5,
                                     double d6,
                                     double d7) {
  return target(d0, d1, d2, d3, d4, d5, d6, d7);
}

// Calls to native Dart functions are based on this interface.
typedef void (*SimulatorNativeCallWrapper)(Dart_NativeArguments arguments,
                                           Dart_NativeFunction target);

void Simulator::DoRedirectedCall(Instr* instr) {
  SimulatorSetjmpBuffer buffer(this);
  if (!setjmp(buffer.buffer_)) {
    int64_t saved_lr = get_register(LR);
    Redirection* redirection = Redirection::FromHltInstruction(instr);
    uword external = redirection->external_function();

    if (redirection->call_kind() == kRuntimeCall) {
      NativeArguments* arguments =
          reinterpret_cast<NativeArguments*>(get_register(R0));
      SimulatorRuntimeCall target =
          reinterpret_cast<SimulatorRuntimeCall>(external);
      target(*arguments);
      // Zap result register from void function.
      set_register(R0, icount_);
      set_register(R1, icount_);
    } else if (redirection->call_kind() == kLeafRuntimeCall) {
      ASSERT((0 <= redirection->argument_count()) &&
             (redirection->argument_count() <= 8));
      SimulatorLeafRuntimeCall target =
          reinterpret_cast<SimulatorLeafRuntimeCall>(external);
      const int64_t r0 = get_register(R0);
      const int64_t r1 = get_register(R1);
      const int64_t r2 = get_register(R2);
      const int64_t r3 = get_register(R3);
      const int64_t r4 = get_register(R4);
      const int64_t r5 = get_register(R5);
      const int64_t r6 = get_register(R6);
      const int64_t r7 = get_register(R7);
      const int64_t res =
          InvokeLeafRuntime(target, r0, r1, r2, r3, r4, r5, r6, r7);
      set_register(R0, res);      // Set returned result from function.
      set_register(R1, icount_);  // Zap unused result register.
    } else if (redirection->call_kind() == kLeafFloatRuntimeCall) {
      ASSERT((0 <= redirection->argument_count()) &&
             (redirection->argument_count() <= 8));
      SimulatorLeafFloatRuntimeCall target =
          reinterpret_cast<SimulatorLeafFloatRuntimeCall>(external);
      const double d0 = bit_cast<double, int64_t>(get_vregisterd(V0, 0));
      const double d1 = bit_cast<double, int64_t>(get_vregisterd(V1, 0));
      const double d2 = bit_cast<double, int64_t>(get_vregisterd(V2, 0));
      const double d3 = bit_cast<double, int64_t>(get_vregisterd(V3, 0));
      const double d4 = bit_cast<double, int64_t>(get_vregisterd(V4, 0));
      const double d5 = bit_cast<double, int64_t>(get_vregisterd(V5, 0));
      const double d6 = bit_cast<double, int64_t>(get_vregisterd(V6, 0));
      const double d7 = bit_cast<double, int64_t>(get_vregisterd(V7, 0));
      const double res =
          InvokeFloatLeafRuntime(target, d0, d1, d2, d3, d4, d5, d6, d7);
      set_vregisterd(V0, 0, bit_cast<int64_t, double>(res));
      set_vregisterd(V0, 1, 0);
    } else {
      ASSERT(redirection->call_kind() == kNativeCallWrapper);
      SimulatorNativeCallWrapper wrapper =
          reinterpret_cast<SimulatorNativeCallWrapper>(external);
      Dart_NativeArguments arguments =
          reinterpret_cast<Dart_NativeArguments>(get_register(R0));
      Dart_NativeFunction target =
          reinterpret_cast<Dart_NativeFunction>(get_register(R1));
      wrapper(arguments, target);
      // Zap result register from void function.
      set_register(R0, icount_);
      set_register(R1, icount_);
    }

    // Zap caller-saved registers, since the actual runtime call could have
    // used them.
    set_register(R2, icount_);
    set_register(R3, icount_);
    set_register(R4, icount_);
    set_register(R5, icount_);
    set_register(R6, icount_);
    set_register(R7, icount_);
    set_register(R8, icount_);
    set_register(R9, icount_);
    set_register(R10, icount_);
    set_register(R11, icount_);
    set_register(R12, icount_);
    set_register(R13, icount_);
    set_register(R14, icount_);
    set_register(R15, icount_);
    set_register(IP0, icount_);
    set_register(IP1, icount_);
    registers_[ZR] = 0;
    set_register(LR, icount_);

    // TODO(zra): Zap caller-saved fpu registers.

    // Return.
    set_pc(saved_lr);
  } else {
    // Coming via long jump from a throw. Continue to exception handler.
  }
}

void Simulator::DecodeExceptionGen(Instr* instr) {
  if ((instr->Bits(0, 2) == 0) && (instr->Bits(2, 3) == 0) &&
      (instr->Bits(21, 3) == 2)) {
    // Format(instr, "hlt 'imm16");
    uint16_t imm = static_cast<uint16_t>(instr->Imm16Field());
    if (imm == Instr::kSimulatorRedirectCode) {
      DoRedirectedCall(instr);
    }
  }
}

void Simulator::DecodeSystem(Instr* instr) {
  if (instr->InstructionBits() == CLREX) {
    // Format(instr, "clrex");
    ClearExclusive();
    return;
  }
}

void Simulator::DecodeTestAndBranch(Instr* instr) {
  const int op = instr->Bit(24);
  const int bitpos = instr->Bits(19, 5) | (instr->Bit(31) << 5);
  const uint64_t imm14 = instr->SImm14Field();
  const uint64_t dest = get_pc() + (imm14 << 2);
  const Register rt = instr->RtField();
  const uint64_t rt_val = get_register(rt);
  if (op == 0) {
    // Format(instr, "tbz'sf 'rt, 'bitpos, 'dest14");
    if ((rt_val & (1ull << bitpos)) == 0) {
      set_pc(dest);
    }
  } else {
    // Format(instr, "tbnz'sf 'rt, 'bitpos, 'dest14");
    if ((rt_val & (1ull << bitpos)) != 0) {
      set_pc(dest);
    }
  }
}

void Simulator::DecodeUnconditionalBranch(Instr* instr) {
  const bool link = instr->Bit(31) == 1;
  const uint64_t imm26 = instr->SImm26Field();
  const uint64_t dest = get_pc() + (imm26 << 2);
  const uint64_t ret = get_pc() + Instr::kInstrSize;
  set_pc(dest);
  if (link) {
    set_register(LR, ret);
  }
}

void Simulator::DecodeUnconditionalBranchReg(Instr* instr) {
  if ((instr->Bits(0, 5) == 0) && (instr->Bits(10, 6) == 0) &&
      (instr->Bits(16, 5) == 0x1f)) {
    switch (instr->Bits(21, 4)) {
      case 0: {
        // Format(instr, "br 'rn");
        const Register rn = instr->RnField();
        const int64_t dest = get_register(rn);
        set_pc(dest);
        break;
      }
      case 1: {
        // Format(instr, "blr 'rn");
        const Register rn = instr->RnField();
        const int64_t dest = get_register(rn);
        const int64_t ret = get_pc() + Instr::kInstrSize;
        set_pc(dest);
        set_register(LR, ret);
        break;
      }
      case 2: {
        // Format(instr, "ret 'rn");
        const Register rn = instr->RnField();
        const int64_t rn_val = get_register(rn);
        set_pc(rn_val);
        break;
      }
      default:
        UnimplementedInstruction(instr);
        break;
    }
  }
}

void Simulator::DecodeCompareBranch(Instr* instr) {
  if (instr->IsCompareAndBranchOp()) {
    DecodeCompareAndBranch(instr);
  } else if (instr->IsConditionalBranchOp()) {
    DecodeConditionalBranch(instr);
  } else if (instr->IsExceptionGenOp()) {
    DecodeExceptionGen(instr);
  } else if (instr->IsSystemOp()) {
    DecodeSystem(instr);
  } else if (instr->IsTestAndBranchOp()) {
    DecodeTestAndBranch(instr);
  } else if (instr->IsUnconditionalBranchOp()) {
    DecodeUnconditionalBranch(instr);
  } else if (instr->IsUnconditionalBranchRegOp()) {
    DecodeUnconditionalBranchReg(instr);
  }
}

void Simulator::DecodeLoadStoreReg(Instr* instr) {
  // Calculate the address.
  const Register rn = instr->RnField();
  const Register rt = instr->RtField();
  const VRegister vt = instr->VtField();
  const int64_t rn_val = get_register(rn);
  const uint32_t size = (instr->Bit(26) == 1)
                            ? ((instr->Bit(23) << 2) | instr->SzField())
                            : instr->SzField();
  uword address = 0;
  uword wb_address = 0;
  bool wb = false;
  if (instr->Bit(24) == 1) {
    // addr = rn + scaled unsigned 12-bit immediate offset.
    const uint32_t imm12 = static_cast<uint32_t>(instr->Imm12Field());
    const uint32_t offset = imm12 << size;
    address = rn_val + offset;
  } else if (instr->Bits(10, 2) == 0) {
    // addr = rn + signed 9-bit immediate offset.
    wb = false;
    const int64_t offset = static_cast<int64_t>(instr->SImm9Field());
    address = rn_val + offset;
    wb_address = rn_val;
  } else if (instr->Bit(10) == 1) {
    // addr = rn + signed 9-bit immediate offset.
    wb = true;
    const int64_t offset = static_cast<int64_t>(instr->SImm9Field());
    if (instr->Bit(11) == 1) {
      // Pre-index.
      address = rn_val + offset;
      wb_address = address;
    } else {
      // Post-index.
      address = rn_val;
      wb_address = rn_val + offset;
    }
  } else if (instr->Bits(10, 2) == 2) {
    // addr = rn + (rm EXT optionally scaled by operand instruction size).
    const Register rm = instr->RmField();
    const Extend ext = instr->ExtendTypeField();
    const uint8_t scale = (ext == UXTX) && (instr->Bit(12) == 1) ? size : 0;
    const int64_t rm_val = get_register(rm);
    const int64_t offset = ExtendOperand(kXRegSizeInBits, rm_val, ext, scale);
    address = rn_val + offset;
  }

  // Do access.
  if (instr->Bit(26) == 1) {
    if (instr->Bit(22) == 0) {
      // Format(instr, "fstr'fsz 'vt, 'memop");
      const int64_t vt_val = get_vregisterd(vt, 0);
      switch (size) {
        case 2:
          WriteW(address, vt_val & kWRegMask, instr);
          break;
        case 3:
          WriteX(address, vt_val, instr);
          break;
        case 4: {
          simd_value_t val;
          get_vregister(vt, &val);
          WriteX(address, val.bits.i64[0], instr);
          WriteX(address + kWordSize, val.bits.i64[1], instr);
          break;
        }
        default:
          UnimplementedInstruction(instr);
          return;
      }
    } else {
      // Format(instr, "fldr'fsz 'vt, 'memop");
      switch (size) {
        case 2:
          set_vregisterd(vt, 0, static_cast<int64_t>(ReadWU(address, instr)));
          set_vregisterd(vt, 1, 0);
          break;
        case 3:
          set_vregisterd(vt, 0, ReadX(address, instr));
          set_vregisterd(vt, 1, 0);
          break;
        case 4: {
          simd_value_t val;
          val.bits.i64[0] = ReadX(address, instr);
          val.bits.i64[1] = ReadX(address + kWordSize, instr);
          set_vregister(vt, val);
          break;
        }
        default:
          UnimplementedInstruction(instr);
          return;
      }
    }
  } else {
    if (instr->Bits(22, 2) == 0) {
      // Format(instr, "str'sz 'rt, 'memop");
      const int32_t rt_val32 = get_wregister(rt);
      switch (size) {
        case 0: {
          const uint8_t val = static_cast<uint8_t>(rt_val32);
          WriteB(address, val);
          break;
        }
        case 1: {
          const uint16_t val = static_cast<uint16_t>(rt_val32);
          WriteH(address, val, instr);
          break;
        }
        case 2: {
          const uint32_t val = static_cast<uint32_t>(rt_val32);
          WriteW(address, val, instr);
          break;
        }
        case 3: {
          const int64_t val = get_register(rt);
          WriteX(address, val, instr);
          break;
        }
        default:
          UNREACHABLE();
          break;
      }
    } else {
      // Format(instr, "ldr'sz 'rt, 'memop");

      // Read the value.
      const bool signd = instr->Bit(23) == 1;
      // Write the W register for signed values when size < 2.
      // Write the W register for unsigned values when size == 2.
      const bool use_w =
          (signd && (instr->Bit(22) == 1)) || (!signd && (size == 2));
      int64_t val = 0;  // Sign extend into an int64_t.
      switch (size) {
        case 0: {
          if (signd) {
            val = static_cast<int64_t>(ReadB(address));
          } else {
            val = static_cast<int64_t>(ReadBU(address));
          }
          break;
        }
        case 1: {
          if (signd) {
            val = static_cast<int64_t>(ReadH(address, instr));
          } else {
            val = static_cast<int64_t>(ReadHU(address, instr));
          }
          break;
        }
        case 2: {
          if (signd) {
            val = static_cast<int64_t>(ReadW(address, instr));
          } else {
            val = static_cast<int64_t>(ReadWU(address, instr));
          }
          break;
        }
        case 3:
          val = ReadX(address, instr);
          break;
        default:
          UNREACHABLE();
          break;
      }

      // Write to register.
      if (use_w) {
        set_wregister(rt, static_cast<int32_t>(val));
      } else {
        set_register(rt, val);
      }
    }
  }

  // Do writeback.
  if (wb) {
    set_register(rn, wb_address);
  }
}

void Simulator::DecodeLoadStoreRegPair(Instr* instr) {
  const int32_t opc = instr->Bits(23, 3);
  const Register rn = instr->RnField();
  const int64_t rn_val = get_register(rn);
  const intptr_t shift =
      (instr->Bit(26) == 1) ? 2 + instr->SzField() : 2 + instr->SFField();
  const intptr_t size = 1 << shift;
  const int32_t offset = (static_cast<uint32_t>(instr->SImm7Field()) << shift);
  uword address = 0;
  uword wb_address = 0;
  bool wb = false;

  // Calculate address.
  switch (opc) {
    case 1:
      address = rn_val;
      wb_address = rn_val + offset;
      wb = true;
      break;
    case 2:
      address = rn_val + offset;
      break;
    case 3:
      address = rn_val + offset;
      wb_address = address;
      wb = true;
      break;
    default:
      UnimplementedInstruction(instr);
      return;
  }

  // Do access.
  if (instr->Bit(26) == 1) {
    // SIMD/FP.
    const VRegister vt = instr->VtField();
    const VRegister vt2 = instr->Vt2Field();
    if (instr->Bit(22)) {
      // Format(instr, "ldp 'vt, 'vt2, 'memop");
      switch (size) {
        case 4:
          set_vregisterd(vt, 0, static_cast<int64_t>(ReadWU(address, instr)));
          set_vregisterd(vt, 1, 0);
          set_vregisterd(vt2, 0,
                         static_cast<int64_t>(ReadWU(address + 4, instr)));
          set_vregisterd(vt2, 1, 0);
          break;
        case 8:
          set_vregisterd(vt, 0, ReadX(address, instr));
          set_vregisterd(vt, 1, 0);
          set_vregisterd(vt2, 0, ReadX(address + 8, instr));
          set_vregisterd(vt2, 1, 0);
          break;
        case 16: {
          simd_value_t val;
          val.bits.i64[0] = ReadX(address, instr);
          val.bits.i64[1] = ReadX(address + 8, instr);
          set_vregister(vt, val);
          val.bits.i64[0] = ReadX(address + 16, instr);
          val.bits.i64[1] = ReadX(address + 24, instr);
          set_vregister(vt2, val);
          break;
        }
        default:
          UnimplementedInstruction(instr);
          return;
      }
    } else {
      // Format(instr, "stp 'vt, 'vt2, 'memop");
      switch (size) {
        case 4:
          WriteW(address, get_vregisterd(vt, 0) & kWRegMask, instr);
          WriteW(address + 4, get_vregisterd(vt2, 0) & kWRegMask, instr);
          break;
        case 8:
          WriteX(address, get_vregisterd(vt, 0), instr);
          WriteX(address + 8, get_vregisterd(vt2, 0), instr);
          break;
        case 16: {
          simd_value_t val;
          get_vregister(vt, &val);
          WriteX(address, val.bits.i64[0], instr);
          WriteX(address + 8, val.bits.i64[1], instr);
          get_vregister(vt2, &val);
          WriteX(address + 16, val.bits.i64[0], instr);
          WriteX(address + 24, val.bits.i64[1], instr);
          break;
        }
        default:
          UnimplementedInstruction(instr);
          return;
      }
    }
  } else {
    // Integer.
    const Register rt = instr->RtField();
    const Register rt2 = instr->Rt2Field();
    if (instr->Bit(22)) {
      // Format(instr, "ldp'sf 'rt, 'rt2, 'memop");
      const bool signd = instr->Bit(30) == 1;
      int64_t val1 = 0;  // Sign extend into an int64_t.
      int64_t val2 = 0;
      if (instr->Bit(31) == 1) {
        // 64-bit read.
        val1 = ReadX(address, instr);
        val2 = ReadX(address + size, instr);
      } else {
        if (signd) {
          val1 = static_cast<int64_t>(ReadW(address, instr));
          val2 = static_cast<int64_t>(ReadW(address + size, instr));
        } else {
          val1 = static_cast<int64_t>(ReadWU(address, instr));
          val2 = static_cast<int64_t>(ReadWU(address + size, instr));
        }
      }
      // Write to register.
      if (instr->Bit(31) == 1) {
        set_register(rt, val1);
        set_register(rt2, val2);
      } else {
        set_wregister(rt, static_cast<int32_t>(val1));
        set_wregister(rt2, static_cast<int32_t>(val2));
      }
    } else {
      // Format(instr, "stp'sf 'rt, 'rt2, 'memop");
      if (instr->Bit(31) == 1) {
        const int64_t val1 = get_register(rt);
        const int64_t val2 = get_register(rt2);
        WriteX(address, val1, instr);
        WriteX(address + size, val2, instr);
      } else {
        const int32_t val1 = get_wregister(rt);
        const int32_t val2 = get_wregister(rt2);
        WriteW(address, val1, instr);
        WriteW(address + size, val2, instr);
      }
    }
  }

  // Do writeback.
  if (wb) {
    set_register(rn, wb_address);
  }
}

void Simulator::DecodeLoadRegLiteral(Instr* instr) {
  const Register rt = instr->RtField();
  const int64_t off = instr->SImm19Field() << 2;
  const int64_t pc = reinterpret_cast<int64_t>(instr);
  const int64_t address = pc + off;
  const int64_t val = ReadX(address, instr);
  if (instr->Bit(30)) {
    // Format(instr, "ldrx 'rt, 'pcldr");
    set_register(rt, val);
  } else {
    // Format(instr, "ldrw 'rt, 'pcldr");
    set_wregister(rt, static_cast<int32_t>(val));
  }
}

void Simulator::DecodeLoadStoreExclusive(Instr* instr) {
  const int32_t size = instr->Bits(30, 2);
  const Register rs = instr->RsField();
  const Register rn = instr->RnField();
  const Register rt = instr->RtField();
  ASSERT(instr->Rt2Field() == R31);  // Should-Be-One
  const bool is_load = instr->Bit(22) == 1;
  const bool is_exclusive = instr->Bit(23) == 0;
  const bool is_ordered = instr->Bit(15) == 1;
  if (is_load) {
    const bool is_load_acquire = !is_exclusive && is_ordered;
    if (is_load_acquire) {
      ASSERT(rs == R31);  // Should-Be-One
      // Format(instr, "ldar 'rt, 'rn");
      const int64_t addr = get_register(rn);
      const intptr_t value =
          (size == 3) ? ReadAcquire(addr, instr) : ReadAcquireW(addr, instr);
      set_register(rt, value);
    } else {
      ASSERT(rs == R31);  // Should-Be-One
      // Format(instr, "ldxr 'rt, 'rn");
      const int64_t addr = get_register(rn);
      const intptr_t value = (size == 3) ? ReadExclusiveX(addr, instr)
                                         : ReadExclusiveW(addr, instr);
      set_register(rt, value);
    }
  } else {
    const bool is_store_release = !is_exclusive && is_ordered;
    if (is_store_release) {
      ASSERT(rs == R31);  // Should-Be-One
      // Format(instr, "stlr 'rt, 'rn");
      const uword value = get_register(rt);
      const uword addr = get_register(rn);
      if (size == 3) {
        WriteRelease(addr, value, instr);
      } else {
        WriteReleaseW(addr, static_cast<uint32_t>(value), instr);
      }
    } else {
      // Format(instr, "stxr 'rs, 'rt, 'rn");
      const uword value = get_register(rt);
      const uword addr = get_register(rn);
      const intptr_t status =
          (size == 3)
              ? WriteExclusiveX(addr, value, instr)
              : WriteExclusiveW(addr, static_cast<uint32_t>(value), instr);
      set_register(rs, status);
    }
  }
}

void Simulator::DecodeLoadStore(Instr* instr) {
  if (instr->IsLoadStoreRegOp()) {
    DecodeLoadStoreReg(instr);
  } else if (instr->IsLoadStoreRegPairOp()) {
    DecodeLoadStoreRegPair(instr);
  } else if (instr->IsLoadRegLiteralOp()) {
    DecodeLoadRegLiteral(instr);
  } else if (instr->IsLoadStoreExclusiveOp()) {
    DecodeLoadStoreExclusive(instr);
  }
}

int64_t Simulator::ShiftOperand(uint8_t reg_size,
                                int64_t value,
                                Shift shift_type,
                                uint8_t amount) {
  if (amount == 0) {
    return value;
  }
  int64_t mask = reg_size == kXRegSizeInBits ? kXRegMask : kWRegMask;
  switch (shift_type) {
    case LSL:
      return (static_cast<uint64_t>(value) << amount) & mask;
    case LSR:
      return static_cast<uint64_t>(value) >> amount;
    case ASR: {
      // Shift used to restore the sign.
      uint8_t s_shift = kXRegSizeInBits - reg_size;
      // Value with its sign restored.
      int64_t s_value = (value << s_shift) >> s_shift;
      return (s_value >> amount) & mask;
    }
    case ROR: {
      if (reg_size == kWRegSizeInBits) {
        value &= kWRegMask;
      }
      return (static_cast<uint64_t>(value) >> amount) |
             ((static_cast<uint64_t>(value) & ((1ULL << amount) - 1ULL))
              << (reg_size - amount));
    }
    default:
      UNIMPLEMENTED();
      return 0;
  }
}

int64_t Simulator::ExtendOperand(uint8_t reg_size,
                                 int64_t value,
                                 Extend extend_type,
                                 uint8_t amount) {
  switch (extend_type) {
    case UXTB:
      value &= 0xff;
      break;
    case UXTH:
      value &= 0xffff;
      break;
    case UXTW:
      value &= 0xffffffff;
      break;
    case SXTB:
      value = static_cast<int64_t>(static_cast<uint64_t>(value) << 56) >> 56;
      break;
    case SXTH:
      value = static_cast<int64_t>(static_cast<uint64_t>(value) << 48) >> 48;
      break;
    case SXTW:
      value = static_cast<int64_t>(static_cast<uint64_t>(value) << 32) >> 32;
      break;
    case UXTX:
    case SXTX:
      break;
    default:
      UNREACHABLE();
      break;
  }
  int64_t mask = (reg_size == kXRegSizeInBits) ? kXRegMask : kWRegMask;
  return (static_cast<uint64_t>(value) << amount) & mask;
}

int64_t Simulator::DecodeShiftExtendOperand(Instr* instr) {
  const Register rm = instr->RmField();
  const int64_t rm_val = get_register(rm);
  const uint8_t size = instr->SFField() ? kXRegSizeInBits : kWRegSizeInBits;
  if (instr->IsShift()) {
    const Shift shift_type = instr->ShiftTypeField();
    const uint8_t shift_amount = instr->Imm6Field();
    return ShiftOperand(size, rm_val, shift_type, shift_amount);
  } else {
    ASSERT(instr->IsExtend());
    const Extend extend_type = instr->ExtendTypeField();
    const uint8_t shift_amount = instr->Imm3Field();
    return ExtendOperand(size, rm_val, extend_type, shift_amount);
  }
  UNREACHABLE();
  return -1;
}

void Simulator::DecodeAddSubShiftExt(Instr* instr) {
  // Format(instr, "add'sf's 'rd, 'rn, 'shift_op");
  // also, sub, cmp, etc.
  const bool addition = (instr->Bit(30) == 0);
  const Register rd = instr->RdField();
  const Register rn = instr->RnField();
  const uint64_t rm_val = DecodeShiftExtendOperand(instr);
  if (instr->SFField()) {
    // 64-bit add.
    const uint64_t rn_val = get_register(rn);
    const uint64_t alu_out = rn_val + (addition ? rm_val : -rm_val);
    set_register(rd, alu_out);
    if (instr->HasS()) {
      SetNZFlagsX(alu_out);
      SetCFlag(CarryFromX(alu_out, rn_val, rm_val, addition));
      SetVFlag(OverflowFromX(alu_out, rn_val, rm_val, addition));
    }
  } else {
    // 32-bit add.
    const uint32_t rn_val = get_wregister(rn);
    uint32_t rm_val32 = static_cast<uint32_t>(rm_val & kWRegMask);
    uint32_t carry_in = 0;
    if (!addition) {
      carry_in = 1;
      rm_val32 = ~rm_val32;
    }
    const uint32_t alu_out = rn_val + rm_val32 + carry_in;
    set_wregister(rd, alu_out);
    if (instr->HasS()) {
      SetNZFlagsW(alu_out);
      SetCFlag(CarryFromW(rn_val, rm_val32, carry_in));
      SetVFlag(OverflowFromW(rn_val, rm_val32, carry_in));
    }
  }
}

void Simulator::DecodeAddSubWithCarry(Instr* instr) {
  // Format(instr, "adc'sf's 'rd, 'rn, 'rm");
  // Format(instr, "sbc'sf's 'rd, 'rn, 'rm");
  const bool addition = (instr->Bit(30) == 0);
  const Register rd = instr->RdField();
  const Register rn = instr->RnField();
  const Register rm = instr->RmField();
  const uint64_t rn_val64 = get_register(rn);
  const uint32_t rn_val32 = get_wregister(rn);
  const uint64_t rm_val64 = get_register(rm);
  uint32_t rm_val32 = get_wregister(rm);
  const uint32_t carry_in = c_flag_ ? 1 : 0;
  if (instr->SFField()) {
    // 64-bit add.
    const uint64_t alu_out =
        rn_val64 + (addition ? rm_val64 : ~rm_val64) + carry_in;
    set_register(rd, alu_out);
    if (instr->HasS()) {
      SetNZFlagsX(alu_out);
      SetCFlag(CarryFromX(alu_out, rn_val64, rm_val64, addition));
      SetVFlag(OverflowFromX(alu_out, rn_val64, rm_val64, addition));
    }
  } else {
    // 32-bit add.
    if (!addition) {
      rm_val32 = ~rm_val32;
    }
    const uint32_t alu_out = rn_val32 + rm_val32 + carry_in;
    set_wregister(rd, alu_out);
    if (instr->HasS()) {
      SetNZFlagsW(alu_out);
      SetCFlag(CarryFromW(rn_val32, rm_val32, carry_in));
      SetVFlag(OverflowFromW(rn_val32, rm_val32, carry_in));
    }
  }
}

void Simulator::DecodeLogicalShift(Instr* instr) {
  const int op = (instr->Bits(29, 2) << 1) | instr->Bit(21);
  const Register rd = instr->RdField();
  const Register rn = instr->RnField();
  const int64_t rn_val = get_register(rn);
  const int64_t rm_val = DecodeShiftExtendOperand(instr);
  int64_t alu_out = 0;
  switch (op) {
    case 0:
      // Format(instr, "and'sf 'rd, 'rn, 'shift_op");
      alu_out = rn_val & rm_val;
      break;
    case 1:
      // Format(instr, "bic'sf 'rd, 'rn, 'shift_op");
      alu_out = rn_val & (~rm_val);
      break;
    case 2:
      // Format(instr, "orr'sf 'rd, 'rn, 'shift_op");
      alu_out = rn_val | rm_val;
      break;
    case 3:
      // Format(instr, "orn'sf 'rd, 'rn, 'shift_op");
      alu_out = rn_val | (~rm_val);
      break;
    case 4:
      // Format(instr, "eor'sf 'rd, 'rn, 'shift_op");
      alu_out = rn_val ^ rm_val;
      break;
    case 5:
      // Format(instr, "eon'sf 'rd, 'rn, 'shift_op");
      alu_out = rn_val ^ (~rm_val);
      break;
    case 6:
      // Format(instr, "and'sfs 'rd, 'rn, 'shift_op");
      alu_out = rn_val & rm_val;
      break;
    case 7:
      // Format(instr, "bic'sfs 'rd, 'rn, 'shift_op");
      alu_out = rn_val & (~rm_val);
      break;
    default:
      UNREACHABLE();
      break;
  }

  // Set flags if ands or bics.
  if ((op == 6) || (op == 7)) {
    if (instr->SFField() == 1) {
      SetNZFlagsX(alu_out);
    } else {
      SetNZFlagsW(alu_out);
    }
    SetCFlag(false);
    SetVFlag(false);
  }

  if (instr->SFField() == 1) {
    set_register(rd, alu_out);
  } else {
    set_wregister(rd, alu_out & kWRegMask);
  }
}

static int64_t divide64(int64_t top, int64_t bottom, bool signd) {
  // ARM64 does not trap on integer division by zero. The destination register
  // is instead set to 0.
  if (bottom == 0) {
    return 0;
  }

  if (signd) {
    // INT_MIN / -1 = INT_MIN.
    if ((top == static_cast<int64_t>(0x8000000000000000LL)) &&
        (bottom == static_cast<int64_t>(0xffffffffffffffffLL))) {
      return static_cast<int64_t>(0x8000000000000000LL);
    } else {
      return top / bottom;
    }
  } else {
    const uint64_t utop = static_cast<uint64_t>(top);
    const uint64_t ubottom = static_cast<uint64_t>(bottom);
    return static_cast<int64_t>(utop / ubottom);
  }
}

static int32_t divide32(int32_t top, int32_t bottom, bool signd) {
  // ARM64 does not trap on integer division by zero. The destination register
  // is instead set to 0.
  if (bottom == 0) {
    return 0;
  }

  if (signd) {
    // INT_MIN / -1 = INT_MIN.
    if ((top == static_cast<int32_t>(0x80000000)) &&
        (bottom == static_cast<int32_t>(0xffffffff))) {
      return static_cast<int32_t>(0x80000000);
    } else {
      return top / bottom;
    }
  } else {
    const uint32_t utop = static_cast<uint32_t>(top);
    const uint32_t ubottom = static_cast<uint32_t>(bottom);
    return static_cast<int32_t>(utop / ubottom);
  }
}

void Simulator::DecodeMiscDP1Source(Instr* instr) {
  const Register rd = instr->RdField();
  const Register rn = instr->RnField();
  const int op = instr->Bits(10, 10);
  const int64_t rn_val64 = get_register(rn);
  const int32_t rn_val32 = get_wregister(rn);
  switch (op) {
    case 4: {
      // Format(instr, "clz'sf 'rd, 'rn");
      if (instr->SFField() == 1) {
        const uint64_t rd_val = Utils::CountLeadingZeros64(rn_val64);
        set_register(rd, rd_val);
      } else {
        const uint32_t rd_val = Utils::CountLeadingZeros32(rn_val32);
        set_wregister(rd, rd_val);
      }
      break;
    }
    case 0: {
      // Format(instr, "rbit'sf 'rd, 'rn");
      if (instr->SFField() == 1) {
        const uint64_t rd_val = Utils::ReverseBits64(rn_val64);
        set_register(rd, rd_val);
      } else {
        const uint32_t rd_val = Utils::ReverseBits32(rn_val32);
        set_wregister(rd, rd_val);
      }
      break;
    }
    default:
      UnimplementedInstruction(instr);
      break;
  }
}

void Simulator::DecodeMiscDP2Source(Instr* instr) {
  const Register rd = instr->RdField();
  const Register rn = instr->RnField();
  const Register rm = instr->RmField();
  const int op = instr->Bits(10, 5);
  const int64_t rn_val64 = get_register(rn);
  const int64_t rm_val64 = get_register(rm);
  const int32_t rn_val32 = get_wregister(rn);
  const int32_t rm_val32 = get_wregister(rm);
  switch (op) {
    case 2:
    case 3: {
      // Format(instr, "udiv'sf 'rd, 'rn, 'rm");
      // Format(instr, "sdiv'sf 'rd, 'rn, 'rm");
      const bool signd = instr->Bit(10) == 1;
      if (instr->SFField() == 1) {
        set_register(rd, divide64(rn_val64, rm_val64, signd));
      } else {
        set_wregister(rd, divide32(rn_val32, rm_val32, signd));
      }
      break;
    }
    case 8: {
      // Format(instr, "lsl'sf 'rd, 'rn, 'rm");
      if (instr->SFField() == 1) {
        const uint64_t rn_u64 = static_cast<uint64_t>(rn_val64);
        const int64_t alu_out = rn_u64 << (rm_val64 & (kXRegSizeInBits - 1));
        set_register(rd, alu_out);
      } else {
        const uint32_t rn_u32 = static_cast<uint32_t>(rn_val32);
        const int32_t alu_out = rn_u32 << (rm_val32 & (kXRegSizeInBits - 1));
        set_wregister(rd, alu_out);
      }
      break;
    }
    case 9: {
      // Format(instr, "lsr'sf 'rd, 'rn, 'rm");
      if (instr->SFField() == 1) {
        const uint64_t rn_u64 = static_cast<uint64_t>(rn_val64);
        const int64_t alu_out = rn_u64 >> (rm_val64 & (kXRegSizeInBits - 1));
        set_register(rd, alu_out);
      } else {
        const uint32_t rn_u32 = static_cast<uint32_t>(rn_val32);
        const int32_t alu_out = rn_u32 >> (rm_val32 & (kXRegSizeInBits - 1));
        set_wregister(rd, alu_out);
      }
      break;
    }
    case 10: {
      // Format(instr, "asr'sf 'rd, 'rn, 'rm");
      if (instr->SFField() == 1) {
        const int64_t alu_out = rn_val64 >> (rm_val64 & (kXRegSizeInBits - 1));
        set_register(rd, alu_out);
      } else {
        const int32_t alu_out = rn_val32 >> (rm_val32 & (kXRegSizeInBits - 1));
        set_wregister(rd, alu_out);
      }
      break;
    }
    default:
      UnimplementedInstruction(instr);
      break;
  }
}

void Simulator::DecodeMiscDP3Source(Instr* instr) {
  const Register rd = instr->RdField();
  const Register rn = instr->RnField();
  const Register rm = instr->RmField();
  const Register ra = instr->RaField();
  if ((instr->Bits(29, 2) == 0) && (instr->Bits(21, 3) == 0) &&
      (instr->Bit(15) == 0)) {
    // Format(instr, "madd'sf 'rd, 'rn, 'rm, 'ra");
    if (instr->SFField() == 1) {
      const uint64_t rn_val = get_register(rn);
      const uint64_t rm_val = get_register(rm);
      const uint64_t ra_val = get_register(ra);
      const uint64_t alu_out = ra_val + (rn_val * rm_val);
      set_register(rd, alu_out);
    } else {
      const uint32_t rn_val = get_wregister(rn);
      const uint32_t rm_val = get_wregister(rm);
      const uint32_t ra_val = get_wregister(ra);
      const uint32_t alu_out = ra_val + (rn_val * rm_val);
      set_wregister(rd, alu_out);
    }
  } else if ((instr->Bits(29, 2) == 0) && (instr->Bits(21, 3) == 0) &&
             (instr->Bit(15) == 1)) {
    // Format(instr, "msub'sf 'rd, 'rn, 'rm, 'ra");
    if (instr->SFField() == 1) {
      const uint64_t rn_val = get_register(rn);
      const uint64_t rm_val = get_register(rm);
      const uint64_t ra_val = get_register(ra);
      const uint64_t alu_out = ra_val - (rn_val * rm_val);
      set_register(rd, alu_out);
    } else {
      const uint32_t rn_val = get_wregister(rn);
      const uint32_t rm_val = get_wregister(rm);
      const uint32_t ra_val = get_wregister(ra);
      const uint32_t alu_out = ra_val - (rn_val * rm_val);
      set_wregister(rd, alu_out);
    }
  } else if ((instr->Bits(29, 3) == 4) && (instr->Bits(21, 3) == 2) &&
             (instr->Bit(15) == 0)) {
    ASSERT(ra == R31);  // Should-Be-One
    // Format(instr, "smulh 'rd, 'rn, 'rm");
    const int64_t rn_val = get_register(rn);
    const int64_t rm_val = get_register(rm);
#if defined(DART_HOST_OS_WINDOWS)
    // Visual Studio does not support __int128.
    int64_t alu_out;
    Multiply128(rn_val, rm_val, &alu_out);
#else
    const __int128 res =
        static_cast<__int128>(rn_val) * static_cast<__int128>(rm_val);
    const int64_t alu_out = static_cast<int64_t>(res >> 64);
#endif  // DART_HOST_OS_WINDOWS
    set_register(rd, alu_out);
  } else if ((instr->Bits(29, 3) == 4) && (instr->Bits(21, 3) == 6) &&
             (instr->Bit(15) == 0)) {
    ASSERT(ra == R31);  // Should-Be-One
    // Format(instr, "umulh 'rd, 'rn, 'rm");
    const uint64_t rn_val = get_register(rn);
    const uint64_t rm_val = get_register(rm);
#if defined(DART_HOST_OS_WINDOWS)
    // Visual Studio does not support __int128.
    uint64_t alu_out;
    UnsignedMultiply128(rn_val, rm_val, &alu_out);
#else
    const unsigned __int128 res = static_cast<unsigned __int128>(rn_val) *
                                  static_cast<unsigned __int128>(rm_val);
    const uint64_t alu_out = static_cast<uint64_t>(res >> 64);
#endif  // DART_HOST_OS_WINDOWS
    set_register(rd, alu_out);
  } else if ((instr->Bits(29, 3) == 4) && (instr->Bit(15) == 0)) {
    if (instr->Bits(21, 3) == 5) {
      // Format(instr, "umaddl 'rd, 'rn, 'rm, 'ra");
      const uint64_t rn_val = static_cast<uint32_t>(get_wregister(rn));
      const uint64_t rm_val = static_cast<uint32_t>(get_wregister(rm));
      const uint64_t ra_val = get_register(ra);
      const uint64_t alu_out = ra_val + (rn_val * rm_val);
      set_register(rd, alu_out);
    } else {
      // Format(instr, "smaddl 'rd, 'rn, 'rm, 'ra");
      const int64_t rn_val = static_cast<int32_t>(get_wregister(rn));
      const int64_t rm_val = static_cast<int32_t>(get_wregister(rm));
      const int64_t ra_val = get_register(ra);
      const int64_t alu_out = ra_val + (rn_val * rm_val);
      set_register(rd, alu_out);
    }
  }
}

void Simulator::DecodeConditionalSelect(Instr* instr) {
  const Register rd = instr->RdField();
  const Register rn = instr->RnField();
  const Register rm = instr->RmField();
  const int64_t rm_val64 = get_register(rm);
  const int32_t rm_val32 = get_wregister(rm);
  const int64_t rn_val64 = get_register(rn);
  const int32_t rn_val32 = get_wregister(rn);
  int64_t result64 = 0;
  int32_t result32 = 0;

  if ((instr->Bits(29, 2) == 0) && (instr->Bits(10, 2) == 0)) {
    // Format(instr, "mov'sf'cond 'rd, 'rn, 'rm");
    result64 = rm_val64;
    result32 = rm_val32;
    if (ConditionallyExecute(instr)) {
      result64 = rn_val64;
      result32 = rn_val32;
    }
  } else if ((instr->Bits(29, 2) == 0) && (instr->Bits(10, 2) == 1)) {
    // Format(instr, "csinc'sf'cond 'rd, 'rn, 'rm");
    result64 = rm_val64 + 1;
    result32 = rm_val32 + 1;
    if (ConditionallyExecute(instr)) {
      result64 = rn_val64;
      result32 = rn_val32;
    }
  } else if ((instr->Bits(29, 2) == 2) && (instr->Bits(10, 2) == 0)) {
    // Format(instr, "csinv'sf'cond 'rd, 'rn, 'rm");
    result64 = ~rm_val64;
    result32 = ~rm_val32;
    if (ConditionallyExecute(instr)) {
      result64 = rn_val64;
      result32 = rn_val32;
    }
  } else if ((instr->Bits(29, 2) == 2) && (instr->Bits(10, 2) == 1)) {
    // Format(instr, "csneg'sf'cond 'rd, 'rn, 'rm");
    result64 = -rm_val64;
    result32 = -rm_val32;
    if (ConditionallyExecute(instr)) {
      result64 = rn_val64;
      result32 = rn_val32;
    }
  }

  if (instr->SFField() == 1) {
    set_register(rd, result64);
  } else {
    set_wregister(rd, result32);
  }
}

void Simulator::DecodeDPRegister(Instr* instr) {
  if (instr->IsAddSubShiftExtOp()) {
    DecodeAddSubShiftExt(instr);
  } else if (instr->IsAddSubWithCarryOp()) {
    DecodeAddSubWithCarry(instr);
  } else if (instr->IsLogicalShiftOp()) {
    DecodeLogicalShift(instr);
  } else if (instr->IsMiscDP1SourceOp()) {
    DecodeMiscDP1Source(instr);
  } else if (instr->IsMiscDP2SourceOp()) {
    DecodeMiscDP2Source(instr);
  } else if (instr->IsMiscDP3SourceOp()) {
    DecodeMiscDP3Source(instr);
  } else if (instr->IsConditionalSelectOp()) {
    DecodeConditionalSelect(instr);
  }
}

void Simulator::DecodeFPImm(Instr* instr) {
  if (instr->Bit(22) == 1) {
    // Double.
    // Format(instr, "fmovd 'vd, #'immd");
    const VRegister vd = instr->VdField();
    const int64_t immd = Instr::VFPExpandImm(instr->Imm8Field());
    set_vregisterd(vd, 0, immd);
    set_vregisterd(vd, 1, 0);
  }
}

void Simulator::DecodeFPIntCvt(Instr* instr) {
  const VRegister vd = instr->VdField();
  const VRegister vn = instr->VnField();
  const Register rd = instr->RdField();
  const Register rn = instr->RnField();

  if ((instr->SFField() == 0) && (instr->Bits(22, 2) == 0)) {
    if (instr->Bits(16, 5) == 6) {
      // Format(instr, "fmovrs'sf 'rd, 'vn");
      const int32_t vn_val = get_vregisters(vn, 0);
      set_wregister(rd, vn_val);
    } else if (instr->Bits(16, 5) == 7) {
      // Format(instr, "fmovsr'sf 'vd, 'rn");
      const int32_t rn_val = get_wregister(rn);
      set_vregisters(vd, 0, rn_val);
      set_vregisters(vd, 1, 0);
      set_vregisters(vd, 2, 0);
      set_vregisters(vd, 3, 0);
    }
  } else if (instr->Bits(22, 2) == 1) {
    if (instr->Bits(16, 5) == 2) {
      // Format(instr, "scvtfd'sf 'vd, 'rn");
      const int64_t rn_val64 = get_register(rn);
      const int32_t rn_val32 = get_wregister(rn);
      const double vn_dbl = (instr->SFField() == 1)
                                ? static_cast<double>(rn_val64)
                                : static_cast<double>(rn_val32);
      set_vregisterd(vd, 0, bit_cast<int64_t, double>(vn_dbl));
      set_vregisterd(vd, 1, 0);
    } else if (instr->Bits(16, 5) == 6) {
      // Format(instr, "fmovrd'sf 'rd, 'vn");
      const int64_t vn_val = get_vregisterd(vn, 0);
      set_register(rd, vn_val);
    } else if (instr->Bits(16, 5) == 7) {
      // Format(instr, "fmovdr'sf 'vd, 'rn");
      const int64_t rn_val = get_register(rn);
      set_vregisterd(vd, 0, rn_val);
      set_vregisterd(vd, 1, 0);
    } else if ((instr->Bits(16, 5) == 8) || (instr->Bits(16, 5) == 16) ||
               (instr->Bits(16, 5) == 24)) {
      const intptr_t max = instr->Bit(31) == 1 ? INT64_MAX : INT32_MAX;
      const intptr_t min = instr->Bit(31) == 1 ? INT64_MIN : INT32_MIN;
      double vn_val = bit_cast<double, int64_t>(get_vregisterd(vn, 0));
      switch (instr->Bits(16, 5)) {
        case 8:
          // Format(instr, "fcvtps'sf 'rd, 'vn");
          vn_val = ceil(vn_val);
          break;
        case 16:
          // Format(instr, "fcvtms'sf 'rd, 'vn");
          vn_val = floor(vn_val);
          break;
        case 24:
          // Format(instr, "fcvtzs'sf 'rd, 'vn");
          break;
      }
      int64_t result;
      if (vn_val >= static_cast<double>(max)) {
        result = max;
      } else if (vn_val <= static_cast<double>(min)) {
        result = min;
      } else {
        result = static_cast<int64_t>(vn_val);
      }
      if (instr->Bit(31) == 1) {
        set_register(rd, result);
      } else {
        set_register(rd, result & 0xffffffffll);
      }
    }
  }
}

void Simulator::DecodeFPOneSource(Instr* instr) {
  const int opc = instr->Bits(15, 6);
  const VRegister vd = instr->VdField();
  const VRegister vn = instr->VnField();
  const int64_t vn_val = get_vregisterd(vn, 0);
  const int32_t vn_val32 = vn_val & kWRegMask;
  const double vn_dbl = bit_cast<double, int64_t>(vn_val);
  const float vn_flt = bit_cast<float, int32_t>(vn_val32);

  int64_t res_val = 0;
  switch (opc) {
    case 0:
      // Format("fmovdd 'vd, 'vn");
      res_val = get_vregisterd(vn, 0);
      break;
    case 1:
      // Format("fabsd 'vd, 'vn");
      res_val = bit_cast<int64_t, double>(fabs(vn_dbl));
      break;
    case 2:
      // Format("fnegd 'vd, 'vn");
      res_val = bit_cast<int64_t, double>(-vn_dbl);
      break;
    case 3:
      // Format("fsqrtd 'vd, 'vn");
      res_val = bit_cast<int64_t, double>(sqrt(vn_dbl));
      break;
    case 4: {
      // Format(instr, "fcvtsd 'vd, 'vn");
      const uint32_t val =
          bit_cast<uint32_t, float>(static_cast<float>(vn_dbl));
      res_val = static_cast<int64_t>(val);
      break;
    }
    case 5:
      // Format(instr, "fcvtds 'vd, 'vn");
      res_val = bit_cast<int64_t, double>(static_cast<double>(vn_flt));
      break;
    default:
      UnimplementedInstruction(instr);
      break;
  }

  set_vregisterd(vd, 0, res_val);
  set_vregisterd(vd, 1, 0);
}

void Simulator::DecodeFPTwoSource(Instr* instr) {
  const VRegister vd = instr->VdField();
  const VRegister vn = instr->VnField();
  const VRegister vm = instr->VmField();
  const double vn_val = bit_cast<double, int64_t>(get_vregisterd(vn, 0));
  const double vm_val = bit_cast<double, int64_t>(get_vregisterd(vm, 0));
  const int opc = instr->Bits(12, 4);
  double result;

  switch (opc) {
    case 0:
      // Format(instr, "fmuld 'vd, 'vn, 'vm");
      result = vn_val * vm_val;
      break;
    case 1:
      // Format(instr, "fdivd 'vd, 'vn, 'vm");
      result = vn_val / vm_val;
      break;
    case 2:
      // Format(instr, "faddd 'vd, 'vn, 'vm");
      result = vn_val + vm_val;
      break;
    case 3:
      // Format(instr, "fsubd 'vd, 'vn, 'vm");
      result = vn_val - vm_val;
      break;
    default:
      UnimplementedInstruction(instr);
      return;
  }

  set_vregisterd(vd, 0, bit_cast<int64_t, double>(result));
  set_vregisterd(vd, 1, 0);
}

void Simulator::DecodeFPCompare(Instr* instr) {
  const VRegister vn = instr->VnField();
  const VRegister vm = instr->VmField();
  const double vn_val = bit_cast<double, int64_t>(get_vregisterd(vn, 0));
  double vm_val;

  if ((instr->Bit(22) == 1) && (instr->Bits(3, 2) == 0)) {
    // Format(instr, "fcmpd 'vn, 'vm");
    vm_val = bit_cast<double, int64_t>(get_vregisterd(vm, 0));
  } else if ((instr->Bit(22) == 1) && (instr->Bits(3, 2) == 1)) {
    if (instr->VmField() == V0) {
      // Format(instr, "fcmpd 'vn, #0.0");
      vm_val = 0.0;
    }
  }

  n_flag_ = false;
  z_flag_ = false;
  c_flag_ = false;
  v_flag_ = false;

  if (isnan(vn_val) || isnan(vm_val)) {
    c_flag_ = true;
    v_flag_ = true;
  } else if (vn_val == vm_val) {
    z_flag_ = true;
    c_flag_ = true;
  } else if (vn_val < vm_val) {
    n_flag_ = true;
  } else {
    c_flag_ = true;
  }
}

void Simulator::DecodeFP(Instr* instr) {
  if (instr->IsFPImmOp()) {
    DecodeFPImm(instr);
  } else if (instr->IsFPIntCvtOp()) {
    DecodeFPIntCvt(instr);
  } else if (instr->IsFPOneSourceOp()) {
    DecodeFPOneSource(instr);
  } else if (instr->IsFPTwoSourceOp()) {
    DecodeFPTwoSource(instr);
  } else if (instr->IsFPCompareOp()) {
    DecodeFPCompare(instr);
  }
}

void Simulator::DecodeDPSimd2(Instr* instr) {
  if (instr->IsFPOp()) {
    DecodeFP(instr);
  }
}

// Executes the current instruction.
void Simulator::InstructionDecode(Instr* instr) {
  pc_modified_ = false;

  if (instr->IsDPImmediateOp()) {
    DecodeDPImmediate(instr);
  } else if (instr->IsCompareBranchOp()) {
    DecodeCompareBranch(instr);
  } else if (instr->IsLoadStoreOp()) {
    DecodeLoadStore(instr);
  } else if (instr->IsDPRegisterOp()) {
    DecodeDPRegister(instr);
  } else if (instr->IsDPSimd2Op()) {
    DecodeDPSimd2(instr);
  }

  if (!pc_modified_) {
    set_pc(reinterpret_cast<int64_t>(instr) + Instr::kInstrSize);
  }
}

void Simulator::Execute() {
  // Get the PC to simulate. Cannot use the accessor here as we need the
  // raw PC value and not the one used as input to arithmetic instructions.
  uword program_counter = get_pc();

  // Fast version of the dispatch loop without checking whether the simulator
  // should be stopping at a particular executed instruction.
  while (program_counter != kEndSimulatingPC) {
    Instr* instr = reinterpret_cast<Instr*>(program_counter);
    InstructionDecode(instr);
    program_counter = get_pc();
  }
}

int64_t Simulator::Call(int64_t entry,
                        int64_t parameter0,
                        int64_t parameter1,
                        int64_t parameter2,
                        int64_t parameter3,
                        bool fp_return,
                        bool fp_args) {
  // Save the SP register before the call so we can restore it.
  const intptr_t sp_before_call = get_register(R31);

  // Setup parameters.
  if (fp_args) {
    set_vregisterd(V0, 0, parameter0);
    set_vregisterd(V0, 1, 0);
    set_vregisterd(V1, 0, parameter1);
    set_vregisterd(V1, 1, 0);
    set_vregisterd(V2, 0, parameter2);
    set_vregisterd(V2, 1, 0);
    set_vregisterd(V3, 0, parameter3);
    set_vregisterd(V3, 1, 0);
  } else {
    set_register(R0, parameter0);
    set_register(R1, parameter1);
    set_register(R2, parameter2);
    set_register(R3, parameter3);
  }

  // Make sure the activation frames are properly aligned.
  intptr_t stack_pointer = sp_before_call;
  if (OS::ActivationFrameAlignment() > 1) {
    stack_pointer =
        Utils::RoundDown(stack_pointer, OS::ActivationFrameAlignment());
  }
  set_register(R31, stack_pointer);

  // Prepare to execute the code at entry.
  set_pc(entry);
  // Put down marker for end of simulation. The simulator will stop simulation
  // when the PC reaches this value. By saving the "end simulation" value into
  // the LR the simulation stops when returning to this call point.
  set_register(LR, kEndSimulatingPC);

  registers_[ZR] = 0;

  // Remember the values of callee-saved registers, and set them up with a
  // known value so that we are able to check that they are preserved
  // properly across Dart execution.
  int64_t preserved_vals[kAbiPreservedCpuRegCount];
  const double dicount = static_cast<double>(icount_);
  const int64_t callee_saved_value = bit_cast<int64_t, double>(dicount);
  for (int i = kAbiFirstPreservedCpuReg; i <= kAbiLastPreservedCpuReg; i++) {
    const Register r = static_cast<Register>(i);
    preserved_vals[i - kAbiFirstPreservedCpuReg] = get_register(r);
    set_register(r, callee_saved_value);
  }

  // Only the bottom half of the V registers must be preserved.
  int64_t preserved_dvals[kAbiPreservedFpuRegCount];
  for (int i = kAbiFirstPreservedFpuReg; i <= kAbiLastPreservedFpuReg; i++) {
    const VRegister r = static_cast<VRegister>(i);
    preserved_dvals[i - kAbiFirstPreservedFpuReg] = get_vregisterd(r, 0);
    set_vregisterd(r, 0, callee_saved_value);
    set_vregisterd(r, 1, 0);
  }

  // Start the simulation.
  Execute();

  // Check that the callee-saved registers have been preserved,
  // and restore them with the original value.
  for (int i = kAbiFirstPreservedCpuReg; i <= kAbiLastPreservedCpuReg; i++) {
    const Register r = static_cast<Register>(i);
    ASSERT(callee_saved_value == get_register(r));
    set_register(r, preserved_vals[i - kAbiFirstPreservedCpuReg]);
  }

  for (int i = kAbiFirstPreservedFpuReg; i <= kAbiLastPreservedFpuReg; i++) {
    const VRegister r = static_cast<VRegister>(i);
    ASSERT(callee_saved_value == get_vregisterd(r, 0));
    set_vregisterd(r, 0, preserved_dvals[i - kAbiFirstPreservedFpuReg]);
    set_vregisterd(r, 1, 0);
  }

  // Restore the SP register and return R0.
  set_register(R31, sp_before_call);
  int64_t return_value;
  if (fp_return) {
    return_value = get_vregisterd(V0, 0);
  } else {
    return_value = get_register(R0);
  }
  return return_value;
}

void Simulator::JumpToFrame(uword pc, uword sp, uword fp, Thread* thread) {
  // Walk over all setjmp buffers (simulated --> C++ transitions)
  // and try to find the setjmp associated with the simulated stack pointer.
  SimulatorSetjmpBuffer* buf = last_setjmp_buffer();
  while (buf->link() != NULL && buf->link()->sp() <= sp) {
    buf = buf->link();
  }
  ASSERT(buf != NULL);

  // The C++ caller has not cleaned up the stack memory of C++ frames.
  // Prepare for unwinding frames by destroying all the stack resources
  // in the previous C++ frames.
  StackResource::Unwind(thread);

  // Keep the following code in sync with `StubCode::JumpToFrameStub()`.

  // Unwind the C++ stack and continue simulation in the target frame.
  set_pc(static_cast<int64_t>(pc));
  set_register(SP, static_cast<int64_t>(sp));
  set_register(FP, static_cast<int64_t>(fp));
  set_register(THR, reinterpret_cast<int64_t>(thread));
  set_register(R31, thread->saved_stack_limit() - 4096);
  // Set the tag.
  thread->set_vm_tag(VMTag::kDartTagId);
  // Clear top exit frame.
  thread->set_top_exit_frame_info(0);
  // Restore pool pointer.
  int64_t code =
      *reinterpret_cast<int64_t*>(fp + kPcMarkerSlotFromFp * kWordSize);
  int64_t pp = FLAG_precompiled_mode
                   ? static_cast<int64_t>(thread->global_object_pool())
                   : *reinterpret_cast<int64_t*>(
                         code + Code::object_pool_offset() - kHeapObjectTag);
  pp -= kHeapObjectTag;  // In the PP register, the pool pointer is untagged.
  set_register(CODE_REG, code);
  set_register(PP, pp);
  set_register(HEAP_BITS, (thread->write_barrier_mask() << 32) |
                              (thread->heap_base() >> 32));
  set_register(NULL_REG, static_cast<int64_t>(Object::null()));
  if (FLAG_precompiled_mode) {
    set_register(DISPATCH_TABLE_REG,
                 reinterpret_cast<int64_t>(thread->dispatch_table_array()));
  }

  buf->Longjmp();
}

}  // namespace dart

#endif  // !defined(USING_SIMULATOR)

#endif  // defined TARGET_ARCH_BD64
