//Groups: @ingroup\s+(API_REF|KNOBS|IMG_BASIC_API|INS_BASIC_API|INS_INST_API|INS_BASIC_API_GEN_IA32|INS_BASIC_API_IA32|INS_MOD_API_GEN_IA32|SEC_BASIC_API|RTN_BASIC_API|REG_BASIC_API|REG_CPU_GENERIC|REG_CPU_IA32|TRACE_BASIC_API|BBL_BASIC_API|SYM_BASIC_API|MISC_PRINT|MISC_PARSE|KNOB_API|KNOB_BASIC|KNOB_PRINT|LOCK|PIN_CONTROL|TRACE_VERSION_API|BUFFER_API|PROTO_API|PIN_PROCESS_API|PIN_THREAD_API|PIN_SYSCALL_API|WINDOWS_SYSCALL_API_UNDOC|DEBUG_API|ERROR_FILE_BASIC|TYPE_BASE|INSTLIB|ALARM|CHILD_PROCESS_API|UTILS|MISC|CONTEXT_API|PHYSICAL_CONTEXT_API|PIN_CALLBACKS|EXCEPTION_API|APPDEBUG_API|STOPPED_THREAD_API|BUFFER_API|PROTO|INST_ARGS|DEPRECATED_PIN_API|INTERNAL_EXCEPTION_PRIVATE_UNDOCUMENTED|PIN_THREAD_PRIVATE|CHILD_PROCESS_INTERNAL|BBL_BASIC|ROGUE_BASIC_API|MESSAGE_TYPE|MESSAGE_BASIC|ERRFILE|MISC_BASIC|ITC_INST_API|CONTEXT_API_UNDOC|EXCEPTION_API_UNDOC|UNDOCUMENTED_PIN_API|OPIN|TRACE_VERSIONS

#ifdef PIN_G_OPERAND_IA32_PH
#error duplicate inclusion of operand_ia32
#else
#define PIN_G_OPERAND_IA32_PH
extern UINT32 INS_OperandCount(INS ins);

extern UINT32 INS_OperandNameId(INS ins, UINT32 n);

extern BOOL INS_OperandIsMemory(INS ins, UINT32 n);

extern REG INS_OperandMemoryBaseReg(INS ins, UINT32 n);

extern REG INS_OperandMemoryIndexReg(INS ins, UINT32 n);

extern REG INS_OperandMemorySegmentReg(INS ins, UINT32 n);

extern UINT32 INS_OperandMemoryScale(INS ins, UINT32 n);

extern ADDRDELTA INS_OperandMemoryDisplacement(INS ins, UINT32 n);

extern BOOL INS_OperandIsFixedMemop(INS ins, UINT32 n);

extern VOID GetNumberAndSizeOfMemAccesses (INS ins, int *numAccesses, int *accessSize, int *indexSize);

extern UINT32 INS_MemoryOperandCount(INS ins);

extern UINT32 INS_MemoryOperandElementCount(INS ins, UINT32 memoryOp);

extern BOOL INS_OperandIsAddressGenerator(INS ins, UINT32 n);

extern BOOL INS_OperandIsBranchDisplacement(INS ins, UINT32 n);

extern BOOL INS_MemoryOperandIsRead(INS ins, UINT32 memopIdx);

extern BOOL INS_MemoryOperandIsWritten(INS ins, UINT32 memopIdx);

extern UINT32 INS_MemoryOperandIndexToOperandIndex(INS ins, UINT32 memopIdx);

extern BOOL INS_OperandIsReg(INS ins, UINT32 n);

extern BOOL INS_OperandIsSegmentReg(INS ins, UINT32 n);

extern REG INS_OperandReg(INS ins, UINT32 n);

extern BOOL INS_OperandIsImmediate(INS ins, UINT32 n);

extern UINT64 INS_OperandImmediate(INS ins, UINT32 n);

extern BOOL INS_OperandIsImplicit(INS ins, UINT32 n);

extern BOOL INS_RegIsImplicit(INS ins, REG reg);

extern UINT32 INS_OperandWidth(INS ins, UINT32 n);

extern BOOL INS_OperandRead(INS ins, UINT32 n);

extern BOOL INS_OperandWritten(INS ins, UINT32 n);

extern BOOL INS_OperandReadOnly(INS ins, UINT32 n);

extern BOOL INS_OperandWrittenOnly(INS ins, UINT32 n);

extern BOOL INS_OperandReadAndWritten(INS ins, UINT32 n);

#endif // PIN_G_OPERAND_IA32_PH


