"""grace hopper ecosystem
grace systems are built around standards certified by Arm to ensure that drivers, opersting systems, applications "just work."
"""
# Compiler Flags: -O3 -mcpu=native 
# -flto to enable link time optimization
# Fortran apps building with -fno-stack-arrays
# -fsigned-char or funsigned-char apps may need this depending on dev assumption
# if fast math optimizations are not acceptable, use -O3 -ffp-contract=fast
# for more accuracy use -fpp-contract=off to disable ploating point operation contraction. 
# Porting apps that use Math Libraries: 
    # NVPL 
    # gcc -DUSE_CBLAS -ffast-math -mcpu=native -O3 \
        # -I/PATH/TO/nvpl/include \
        # -L/PATH/TO/nvpl/lib \
        # -o mt-dgemm.nvpl mt-dgemm.c
        # -lnvpl_blas_ip64_gomp

    # ARMPL
        # gcc -DUSE_CBLAS -ffast-math -mcpu=native -O3 \
            # -l/opt/arm/armpl-23.10.0_Ubuntu-22.04_gcc/include \
            # -L/opt/arm/armpl-23.10.0_Ubuntu-22.04_gcc/lib \
            # -o mt-dgemm.armpl mt-0dgemm.c \
            # -larmpl_lp64

# ATLAS, OpenBLAS, BLIS, ... Community supported with some optimizations for Neoverse V2.
    # Works on Grace, but unlikley to outperform NVPL and ArmPL. A good compatibility option. 
