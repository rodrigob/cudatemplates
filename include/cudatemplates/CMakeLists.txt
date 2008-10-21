add_subdirectory(cublas)
add_subdirectory(specializations)

install(
  DIRECTORY cublas
  DESTINATION include/cudatemplates
  FILES_MATCHING PATTERN "*.hpp"
  PATTERN ".svn" EXCLUDE
)
