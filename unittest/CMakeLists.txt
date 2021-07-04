include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
)

FetchContent_MakeAvailable(googletest)

include(GoogleTest)
ADD_EXECUTABLE(FusionTest
  public/FusionTest.cpp
)

TARGET_LINK_LIBRARIES(FusionTest PRIVATE DeviceFunctions gtest_main)