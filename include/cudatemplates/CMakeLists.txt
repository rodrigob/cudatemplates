add_subdirectory(cublas)
add_subdirectory(auto)

install(
  DIRECTORY cublas
  DESTINATION include/cudatemplates
  FILES_MATCHING PATTERN "*.hpp"
  PATTERN ".svn" EXCLUDE
)
