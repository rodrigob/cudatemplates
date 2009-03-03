#!/usr/bin/perl

@templates = (
    "HostMemoryHeap",
    "HostMemoryLocked",
    "DeviceMemoryLinear",
    "DeviceMemoryPitched",
    "Array"
    );

# no DeviceMemoryPitched and Array in 1d:
sub no_support_1d
{
    return (@_[0] == 3);
}

# only power-of-two size allowed in 3D:
sub only_pot_3d
{
    return (@_[0] == 2) || (@_[0] == 4);
}

sub create_code
{
    my $dim = @_[0];
    open(cpp, "> test${dim}d.cpp");

    foreach $i (0 .. $#templates) {
	foreach $j (0 .. $#templates) {
	    # handle a few exceptions for cases which are not yet implemented:
	    next if ($dim == 1) && (no_support_1d($i) || no_support_1d($j));
	    $smax = "smax${dim}";
	    $smax = 0 if ($dim == 3) && (only_pot_3d($i) || only_pot_3d($j));
	    
	    if(($i == 0) && ($j == 0)) {
		print cpp
		    "err |= test_array_copy2<Cuda::@templates[$i]<float, $dim>, Cuda::@templates[$j]<double, $dim> >".
		    "(size${dim}a, size${dim}b, pos${dim}a, pos${dim}b, size${dim}, $smax);\n".
		    "err |= test_array_copy2<Cuda::@templates[$i]<double, $dim>, Cuda::@templates[$j]<float, $dim> >".
		    "(size${dim}a, size${dim}b, pos${dim}a, pos${dim}b, size${dim}, $smax);\n";
	    }
		
	    print cpp
		"err |= test_array_copy2<Cuda::@templates[$i]<float, $dim>, Cuda::@templates[$j]<float, $dim> >".
		"(size${dim}a, size${dim}b, pos${dim}a, pos${dim}b, size${dim}, $smax);\n";
	}
    }
}

create_code(1);
create_code(2);
create_code(3);
