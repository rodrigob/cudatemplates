install(
  DIRECTORY cudatemplates
  DESTINATION include
  FILES_MATCHING PATTERN "*.hpp"
  PATTERN ".svn" EXCLUDE
)
