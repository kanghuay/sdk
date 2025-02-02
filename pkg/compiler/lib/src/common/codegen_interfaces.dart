// Copyright (c) 2022, the Dart project authors.  Please see the AUTHORS file
// for details. All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

import '../elements/entities.dart';
import '../elements/types.dart' show DartType, InterfaceType;
import '../js/js.dart' as js;
import '../native/behavior.dart';
import '../serialization/serialization.dart';
import '../universe/feature.dart';
import '../universe/selector.dart';
import '../universe/use.dart' show ConstantUse, DynamicUse, StaticUse, TypeUse;
import '../universe/world_impact.dart';
import '../util/util.dart';
import 'codegen_migrated.dart';

abstract class CodegenImpact extends WorldImpact {
  Iterable<Pair<DartType, DartType>> get typeVariableBoundsSubtypeChecks;

  Iterable<Set<ClassEntity>> get specializedGetInterceptors;

  bool get usesInterceptor;

  Iterable<AsyncMarker> get asyncMarkers;

  Iterable<GenericInstantiation> get genericInstantiations;

  Iterable<NativeBehavior> get nativeBehaviors;

  Iterable<FunctionEntity> get nativeMethods;

  Iterable<Selector> get oneShotInterceptors;
}

abstract class CodegenRegistry {
  @deprecated
  void registerInstantiatedClass(ClassEntity element);

  void registerStaticUse(StaticUse staticUse);

  void registerDynamicUse(DynamicUse dynamicUse);

  void registerTypeUse(TypeUse typeUse);

  void registerConstantUse(ConstantUse constantUse);

  void registerTypeVariableBoundsSubtypeCheck(
      DartType subtype, DartType supertype);

  void registerInstantiatedClosure(FunctionEntity element);

  void registerConstSymbol(String name);

  void registerSpecializedGetInterceptor(Set<ClassEntity> classes);

  void registerOneShotInterceptor(Selector selector);

  void registerUseInterceptor();

  void registerInstantiation(InterfaceType type);

  void registerAsyncMarker(AsyncMarker asyncMarker);

  void registerGenericInstantiation(GenericInstantiation instantiation);

  void registerNativeBehavior(NativeBehavior nativeBehavior);

  void registerNativeMethod(FunctionEntity function);

  void registerModularName(ModularName name);

  void registerModularExpression(ModularExpression expression);

  CodegenResult close(js.Fun code);
}

abstract class ModularExpression {}

abstract class CodegenResult {
  void writeToDataSink(DataSinkWriter sink);
}
