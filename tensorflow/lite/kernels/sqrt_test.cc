/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cmath>
#include <initializer_list>
#include <limits>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAre;

// Sqrt kernel registration hook from elementwise op pack
TfLiteRegistration* Register_SQRT();

class SqrtOpModel : public SingleOpModel {
 public:
  SqrtOpModel(BuiltinOperator op, std::initializer_list<int> input_shape) {
    input_ = AddInput(TensorType_FLOAT32);
    output_ = AddOutput(TensorType_FLOAT32);
    SetBuiltinOp(op, BuiltinOptions_NONE, 0);
    resolver_ = std::make_unique<SingleOpResolver>(BuiltinOperator_SQRT,
                                                   Register_SQRT());
    BuildInterpreter({input_shape});
  }

  void SetInput(const std::vector<float>& data) {
    PopulateTensor<float>(input_, data);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 protected:
  int input_;
  int output_;
};

// Regression test for GitHub Issue #118527: TFLite's float32 Sqrt activation
// must cleanly return +inf for positive infinity inputs matching Keras behavior, 
// rather than dropping into vectorized compiler routine integer overflow NaN states.
TEST(SqrtTest, PositiveInfinityReturnsPositiveInfinity) {
  SqrtOpModel m(BuiltinOperator_SQRT, {1, 4});
  const float inf = std::numeric_limits<float>::infinity();
  
  m.SetInput({inf, 9.0f, 16.0f, inf});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  
  // Using native element-wise matchers for structural safety in remote CI environments
  EXPECT_THAT(m.GetOutput(), ElementsAre(
      inf,
      ::testing::FloatNear(3.0f, 1e-6),
      ::testing::FloatNear(4.0f, 1e-6),
      inf
  ));
}

}  // namespace
}  // namespace tflite
