add_subdirectory(cublas)

install(
  DIRECTORY cublas
  DESTINATION include/cudatemplates
  FILES_MATCHING PATTERN "*.hpp"
  PATTERN ".svn" EXCLUDE
)
