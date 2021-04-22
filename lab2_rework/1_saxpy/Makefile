EXECUTABLE := cudaSaxpy
EXECUTABLE_UVM := cudaSaxpyUvm
EXECUTABLE_STREAM := cudaSaxpyStream
CU_FILES   := saxpy_basic.cu saxpy_uvm.cu saxpy_stream.cu 
CU_DEPS    :=
CC_FILES   := main.cpp

DEBUG_ENABLED=0
UVM_ENABLED=0
###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')
CUDA_PATH=$(TACC_CUDA_DIR)
OBJDIR=objs
INC_FLAGS=-I../common
CXX=g++ 
CXXFLAGS=-O0  $(INC_FLAGS) #-Wall
ifeq ($(DEBUG_ENABLED),1)
CXXFLAGS += -g -DVERBOSE # for debugging
endif
ifeq ($(UVM_ENABLED),1)
CXXFLAGS += -DUVM_ENABLED
endif
################################################################################
# When compiling with NVCC, the arch flag (-arch) specifies the name of 
# the NVIDIA GPU architecture that the CUDA files will be compiled for.
# Gencodes (-gencode) allows for more PTX generations, and can be repeated 
# many times for different architectures.
# Pascal (CUDA 8 and later)
# 	SM60 or SM_60, compute_60 – Quadro GP100, Tesla P100, DGX-1 (Generic Pascal)
# 	SM61 or SM_61, compute_61 – GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030, Titan Xp, Tesla P40, Tesla P4
# Volta (CUDA 9 and later)
# 	SM70 or SM_70, compute_70 – DGX-1 with Volta, Tesla V100, GTX 1180 (GV104), Titan V, Quadro GV100
# 	SM72 or SM_72, compute_72 – Jetson AGX Xavier
#
# CUDA code generation flags
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
GENCODE_SM50    := -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 
GENCODE_SM60    := -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61
GENCODE_SM70    := -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70
GENCODE_FLAGS   := $(GENCODE_SM60) # For Pascal architecture
################################################################################
NVCC_LDFLAGS=-L$(CUDA_PATH)/lib64/ -lcudart -Wl,-rpath=$(CUDA_PATH)/lib64
NVCC=$(CUDA_PATH)/bin/nvcc -ccbin=$(CXX) 
NVCC_FLAGS=-O3 -m64 $(GENCODE_FLAGS) $(INC_FLAGS)


COMMON_OBJS=$(OBJDIR)/main.o 
OBJS_BASIC=$(COMMON_OBJS) $(OBJDIR)/saxpy_basic.o 
OBJS_STREAM=$(COMMON_OBJS) $(OBJDIR)/saxpy_stream.o 
OBJS_UVM=$(COMMON_OBJS) $(OBJDIR)/saxpy_uvm.o 


.PHONY: dirs clean

default: $(EXECUTABLE) $(EXECUTABLE_UVM) $(EXECUTABLE_STREAM)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ $(EXECUTABLE) $(EXECUTABLE_UVM) $(EXECUTABLE_STREAM)

$(EXECUTABLE): dirs $(OBJS_BASIC)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS_BASIC) $(NVCC_LDFLAGS)

$(EXECUTABLE_UVM): dirs $(OBJS_UVM)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS_UVM) $(NVCC_LDFLAGS)

$(EXECUTABLE_STREAM): dirs $(OBJS_STREAM)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS_STREAM) $(NVCC_LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCC_FLAGS) -c -o $@
