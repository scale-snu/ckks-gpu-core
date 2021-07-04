include(FetchContent)

FetchContent_Declare(
    rmm
    GIT_REPOSITORY https://github.com/rapidsai/rmm
    GIT_TAG        178c2cd2933fa80b70e69863ba727d19f951a551
)

option(BUILD_TESTS OFF)
FetchContent_MakeAvailable(rmm)