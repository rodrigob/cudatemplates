add_subdirectory(cudatemplates)

install(
  DIRECTORY cudatemplates
  DESTINATION include
  FILES_MATCHING PATTERN "*.hpp"
  PATTERN ".svn" EXCLUDE
)
