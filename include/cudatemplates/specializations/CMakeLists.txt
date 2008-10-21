macro(specialize Name Dim)
  set(Name ${ARGV0})
  set(Dim ${ARGV1})
  math(EXPR DimMax "${Dim} - 1")

  foreach(i RANGE 0 ${DimMax})
    if(i)
      set(SizeArgDecl "${SizeArgDecl}, size_t s${i}")
      set(SizeArgs "${SizeArgs}, s${i}")
    else(i)
      set(SizeArgDecl "size_t s${i}")
      set(SizeArgs "s${i}")
    endif(i)
  endforeach(i)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/specialize_dim.hpp.in ${CMAKE_CURRENT_SOURCE_DIR}/hostmemoryheap${Dim}d.hpp)
endmacro(specialize)

specialize(HostMemoryHeap 1)
specialize(HostMemoryHeap 2)
specialize(HostMemoryHeap 3)

#set(Name HostMemoryHeap)
#set(Dim 1)

#configure_file(specialize_dim.hpp.in hostmemoryheap1d.hpp)

#install(FILES specializations/hostmemoryheap1d.hpp DESTINATION include)
#add_custom_target(specializations DEPENDS specializations/hostmemoryheap1d.hpp)
