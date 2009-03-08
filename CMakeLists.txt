project(cudatemplates CXX C)
set(CMAKE_BUILD_TYPE Debug)
cmake_minimum_required(VERSION 2.6.0)  # FindBoost.cmake is broken in older versions

# testing:
include(CTest)
enable_testing()

set(CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})

set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "2")
set(CPACK_PACKAGE_VERSION_PATCH "2")
set(CPACK_PACKAGE_NAME "cudatemplates")
set(CPACK_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH})

add_subdirectory(include)
add_subdirectory(testing)

configure_file(${CMAKE_SOURCE_DIR}/${CPACK_PACKAGE_NAME}.spec.in ${CMAKE_SOURCE_DIR}/${CPACK_PACKAGE_NAME}.spec)
configure_file(${CMAKE_SOURCE_DIR}/Doxyfile.in ${CMAKE_SOURCE_DIR}/Doxyfile)
configure_file(${CMAKE_SOURCE_DIR}/CudaTemplatesConfigVersion.cmake.in ${CMAKE_SOURCE_DIR}/CudaTemplatesConfigVersion.cmake @ONLY)

if(WIN32)
  install(FILES CudaTemplatesConfig.cmake CudaTemplatesConfigVersion.cmake DESTINATION .)
else(WIN32)
  install(FILES CudaTemplatesConfig.cmake CudaTemplatesConfigVersion.cmake DESTINATION share/CudaTemplates)
endif(WIN32)

set(CPACK_IGNORE_FILES
  ".depend"
  "/#.*#$"
  "/a.out/"
  "/doc/"
  "/src/"
  "/log"
  "CMakeCache\\\\.txt"
  "CMakeFiles"
  "CPack.*Config\\\\.cmake"
  "CVS"
  "Makefile"
  "\\\\.#"
  "\\\\.a$"
  "\\\\.aux$"
  "\\\\.bz2$"
  "\\\\.dsp$"
  "\\\\.dvi$"
  "\\\\.eps$"
  "\\\\.gz$"
  "\\\\.iv$"
  "\\\\.log$"
  "\\\\.o$"
  "\\\\.out"
  "\\\\.pdf$"
  "\\\\.pgm$"
  "\\\\.pro$"
  "\\\\.ps$"
  "\\\\.save"
  "\\\\.so"
  "\\\\.svn"
  "\\\\.swp$"
  "\\\\.tbz$"
  "\\\\.tex$"
  "\\\\.tgz$"
  "^/box.cpp"
  "_CPack_Packages"
  "build"
  "cgparams$"
  "cmake_install.cmake"
  "cvs_import"
  "install.sh"
  "install_manifest.txt"
  "makedist.sh"
  "moc_"
  "progress.make"
  "Testing"
  "~$"
)

set(CPACK_SOURCE_IGNORE_FILES ${CPACK_IGNORE_FILES})
set(CPACK_SOURCE_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION})

if(UNIX)
  set(CPACK_GENERATOR "TBZ2")
  set(CPACK_SOURCE_GENERATOR "TBZ2")
  include(CPack)
endif(UNIX)

add_custom_target(dist COMMAND make package_source)
add_custom_target(doc COMMAND doxygen)
add_custom_target(clean-all COMMAND find \\\( -name CMakeFiles -or -name CMakeCache.txt -or -name Makefile -or -name '*.cmake' -or -name 'CPack*.cmake' -or -name _CPack_Packages -or -name '*~' \\\) -prune -exec rm -rf {} '\;')
