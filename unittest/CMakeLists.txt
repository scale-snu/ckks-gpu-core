include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_TAG main
  GIT_REPOSITORY https://github.com/google/googletest.git
)

FetchContent_MakeAvailable(googletest)

include(GoogleTest)

ADD_EXECUTABLE(FusionTest
  public/FusionTest.cpp
)
ADD_EXECUTABLE(E2ETest
  public/E2ETest.cpp
)
TARGET_LINK_LIBRARIES(FusionTest PRIVATE DeviceFunctions gtest_main)
TARGET_LINK_LIBRARIES(E2ETest PRIVATE DeviceFunctions gtest_main)