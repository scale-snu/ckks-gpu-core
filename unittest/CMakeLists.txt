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
target_include_directories(E2ETest PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/SEAL/install/include/SEAL-4.1/)
TARGET_LINK_LIBRARIES(FusionTest PRIVATE DeviceFunctions gtest_main)
TARGET_LINK_LIBRARIES(E2ETest PRIVATE DeviceFunctions gtest_main  ${CMAKE_SOURCE_DIR}/thirdparty/SEAL/install/lib/libseal-4.1.a)