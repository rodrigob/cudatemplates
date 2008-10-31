set(CTEST_PROJECT_NAME "cudatemplates")
set(CTEST_NIGHTLY_START_TIME "04:00:00 CET")

set(CTEST_DROP_METHOD "http")
set(CTEST_DROP_SITE "cdash.icg.tugraz.at")
set(CTEST_DROP_LOCATION "/submit.php?project=${CTEST_PROJECT_NAME}")
set(CTEST_DROP_SITE_CDASH TRUE)

set(CTEST_SOURCE_DIRECTORY ${CTEST_SCRIPT_DIRECTORY})
set(CTEST_BINARY_DIRECTORY ${CTEST_SCRIPT_DIRECTORY})

set(CTEST_SVN_COMMAND "/usr/bin/svn")
set(CTEST_SVN_CHECKOUT "${CTEST_SVN_COMMAND} co http://svn.w42.at/svn/stock/trunk")
set($ENV{LC_MESSAGES} "en_EN")

# which ctest command to use for running the dashboard
set(CTEST_COMMAND "/usr/bin/ctest -D Nightly")
#set(CTEST_COMMAND "/usr/bin/ctest -D Experimental")

# what cmake command to use for configuring this dashboard
set(CTEST_CMAKE_COMMAND "/usr/bin/cmake")

set(CTEST_ENVIRONMENT "CMAKE_ROOT=/usr/share/cmake")
