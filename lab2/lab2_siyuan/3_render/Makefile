EXECUTABLE := render
CU_FILES   := cudaRenderer.cu
CU_DEPS    :=
CC_FILES   := main.cpp \
              benchmark.cpp refRenderer.cpp \
              noise.cpp ppm.cpp \
              sceneLoader.cpp 

LOGS	   := logs

OPENGL_ENABLED=0
###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')
CUDA_PATH=$(TACC_CUDA_DIR)
OBJDIR=objs
INC_FLAGS=-I../common
CXX=g++
CXXFLAGS=-O3 -Wall $(INC_FLAGS)
LDFLAGS =
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

COMMON_OBJS=$(OBJDIR)/main.o \
			$(OBJDIR)/benchmark.o \
			$(OBJDIR)/refRenderer.o \
			$(OBJDIR)/noise.o \
			$(OBJDIR)/ppm.o \
			$(OBJDIR)/sceneLoader.o 

ifeq ($(OPENGL_ENABLED),1)
CONDA_PATH=$(HOME)/miniconda3
CXXFLAGS += -DENABLE_OPENGL -I$(CONDA_PATH)/include
LDFLAGS += -L$(CONDA_PATH)/lib -lGL -lGLU -lglut -Wl,-rpath=$(CONDA_PATH)/lib
CC_FILES += display.cpp 
COMMON_OBJS += $(OBJDIR)/display.o 
endif

OBJS = $(COMMON_OBJS) $(OBJDIR)/cudaRenderer.o 

.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE) $(LOGS)

check:	default
		./checker.pl

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(NVCC_LDFLAGS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@ $(LDFLAGS)

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCC_FLAGS) -c -o $@
