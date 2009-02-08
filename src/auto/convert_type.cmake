set(filename "convert_type.cu")
file(READ "convert_type_begin.cu" begin)
file(WRITE ${filename} ${begin})
file(READ "convert_type_template.cu" template)

set(types
  "char"
  "unsigned char"
  "short"
  "unsigned short"
  "int"
  "unsigned int"
  "float"
)

foreach(dim 1 2 3)
  math(EXPR dim_asc "119 + ${dim}")
  math(EXPR dim1 "${dim} - 1")
  math(EXPR dim2 "${dim} - 2")
  string(ASCII ${dim_asc} dim_chr)
  set(coords "${coords}\n  int ${dim_chr} = threadIdx.${dim_chr} + blockIdx.${dim_chr} * blockDim.${dim_chr};")
  if(${dim} EQUAL 1)
    set(ofs_src "${dim_chr}")
    set(ofs_dst "${dim_chr}")
  else()
    set(ofs_src "${ofs_src} + ${dim_chr} * src.stride[${dim2}]")
    set(ofs_dst "${ofs_dst} + ${dim_chr} * dst.stride[${dim2}]")
    set(cond "${cond} && ")
  endif()
  set(cond "${cond}${dim_chr} < dst.size[${dim1}]")
  foreach(type1 ${types})
    string(REPLACE " " "_" type1_ ${type1})
    foreach(type2 ${types})
      string(REPLACE " " "_" type2_ ${type2})
      if(NOT ${type1} STREQUAL ${type2})
	string(CONFIGURE "${template}" code @ONLY)
	file(APPEND ${filename} "${code}")
      endif()
    endforeach()
  endforeach()
endforeach()
