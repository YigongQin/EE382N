//Groups: @ingroup\s+(API_REF|KNOBS|IMG_BASIC_API|INS_BASIC_API|INS_INST_API|INS_BASIC_API_GEN_IA32|INS_BASIC_API_IA32|INS_MOD_API_GEN_IA32|SEC_BASIC_API|RTN_BASIC_API|REG_BASIC_API|REG_CPU_GENERIC|REG_CPU_IA32|TRACE_BASIC_API|BBL_BASIC_API|SYM_BASIC_API|MISC_PRINT|MISC_PARSE|KNOB_API|KNOB_BASIC|KNOB_PRINT|LOCK|PIN_CONTROL|TRACE_VERSION_API|BUFFER_API|PROTO_API|PIN_PROCESS_API|PIN_THREAD_API|PIN_SYSCALL_API|WINDOWS_SYSCALL_API_UNDOC|DEBUG_API|ERROR_FILE_BASIC|TYPE_BASE|INSTLIB|ALARM|CHILD_PROCESS_API|UTILS|MISC|CONTEXT_API|PHYSICAL_CONTEXT_API|PIN_CALLBACKS|EXCEPTION_API|APPDEBUG_API|STOPPED_THREAD_API|BUFFER_API|PROTO|INST_ARGS|DEPRECATED_PIN_API|INTERNAL_EXCEPTION_PRIVATE_UNDOCUMENTED|PIN_THREAD_PRIVATE|CHILD_PROCESS_INTERNAL|BBL_BASIC|ROGUE_BASIC_API|MESSAGE_TYPE|MESSAGE_BASIC|ERRFILE|MISC_BASIC|ITC_INST_API|CONTEXT_API_UNDOC|EXCEPTION_API_UNDOC|UNDOCUMENTED_PIN_API|OPIN|TRACE_VERSIONS

#ifdef PIN_G_INS_API_XED_IA32_PH
#error duplicate inclusion of ins_api_xed_ia32
#else
#define PIN_G_INS_API_XED_IA32_PH
extern BOOL INS_HasRealRep(INS ins);

extern BOOL INS_IsStandardMemop(INS ins);

extern BOOL INS_IsCacheLineFlush(INS ins);

extern INT32 INS_Category(const INS ins);

extern INT32 INS_Extension(const INS ins);

extern UINT32 INS_EffectiveAddressWidth(INS ins);

extern USIZE INS_MemoryOperandSize(INS ins, UINT32 memoryOp);

extern USIZE INS_MemoryOperandElementSize(INS ins, UINT32 memoryOp);

extern PIN_DEPRECATED_API USIZE INS_MemoryWriteSize(INS ins);

extern PIN_DEPRECATED_API USIZE INS_MemoryReadSize(INS ins);

extern PREDICATE INS_GetPredicate(INS ins);

extern BOOL INS_IsMemoryRead(INS ins);

extern BOOL INS_IsMemoryWrite(INS ins);

extern BOOL INS_HasMemoryRead2(INS ins);

extern BOOL INS_HasFallThrough(INS ins);

extern BOOL INS_IsSysenter(INS ins);

extern BOOL INS_IsXbegin(INS ins);

extern BOOL INS_IsXend(INS ins);

extern BOOL INS_IsLea(INS ins);

extern BOOL INS_IsNop(INS ins);

extern std::string OPCODE_StringShort(UINT32 opcode);

extern std::string INS_Mnemonic(INS ins);

extern BOOL INS_IsBranch(INS ins);

extern BOOL INS_IsDirectBranch(INS ins);

extern BOOL INS_IsDirectCall(INS ins);

extern PIN_DEPRECATED_API BOOL INS_IsDirectBranchOrCall(INS ins);

extern BOOL INS_IsDirectControlFlow(INS ins);

extern BOOL INS_IsHalt(INS ins);

extern PIN_DEPRECATED_API BOOL INS_IsBranchOrCall(INS ins);

extern BOOL INS_Stutters(INS ins);

extern BOOL INS_IsPcMaterialization(INS ins);

extern BOOL INS_IsCall(INS ins);

extern BOOL INS_IsControlFlow(INS ins);

extern BOOL INS_IsFarCall(INS ins);

extern BOOL INS_IsFarJump(INS ins);

extern BOOL INS_IsDirectFarJump(INS ins);

extern BOOL INS_IsValidForIpointAfter(INS ins);

extern BOOL INS_IsValidForIpointTakenBranch(INS ins);

extern BOOL INS_IsVgather(INS ins);

extern BOOL INS_IsVscatter(INS ins);

extern BOOL INS_HasMemoryVector(INS ins);

extern BOOL INS_HasScatteredMemoryAccess(INS ins);

extern BOOL INS_IsAmx(INS ins);

extern VOID INS_GetFarPointer(INS ins,
                                 UINT16& segment_selector,
                                 UINT32& displacement);

extern BOOL INS_IsProcedureCall(INS ins);

extern BOOL INS_IsInterrupt(INS ins);

extern BOOL INS_IsRet(INS ins);

extern BOOL INS_IsSysret(INS ins);

extern BOOL INS_IsFarRet(INS ins);

extern BOOL INS_IsPrefetch(INS ins);

extern BOOL INS_IsSub(const INS ins);

extern BOOL INS_IsMov(const INS ins);

extern BOOL INS_IsMovFullRegRegSame(const INS ins);

extern BOOL INS_IsAtomicUpdate(const INS ins);

extern BOOL INS_IsRDTSC(const INS ins);

extern BOOL INS_IsMaskMov(const INS ins);

extern REG INS_RepCountRegister(INS ins);

extern PIN_DEPRECATED_API BOOL INS_IsIndirectBranchOrCall(INS ins);

extern BOOL INS_IsIndirectControlFlow(INS ins);

extern REG INS_SegmentRegPrefix(INS ins);

extern BOOL INS_HasExplicitMemoryReference(INS ins);

extern VOID PIN_SetSyntaxIntel();

extern VOID PIN_SetSyntaxATT();

extern VOID PIN_SetSyntaxXED();

extern REG INS_RegR(INS x, UINT32 k);

extern REG INS_RegW(INS x, UINT32 k);

extern OPCODE INS_Opcode(INS ins);

extern std::string CATEGORY_StringShort(UINT32 num);

extern std::string EXTENSION_StringShort(UINT32 num);

extern BOOL INS_SegPrefixIsMemoryRead(INS ins);

extern BOOL INS_SegPrefixIsMemoryWrite(INS ins);

extern BOOL INS_AddressSizePrefix(INS ins);

extern BOOL INS_BranchNotTakenPrefix(INS ins);

extern BOOL INS_BranchTakenPrefix(INS ins);

extern BOOL INS_LockPrefix(INS ins);

extern BOOL INS_OperandSizePrefix(INS ins);

extern BOOL INS_RepPrefix(INS ins);

extern BOOL INS_RepnePrefix(INS ins);

extern BOOL INS_SegmentPrefix(INS ins);

extern BOOL INS_IsXchg(INS ins);

extern BOOL INS_IsStringop(INS ins);

extern BOOL INS_IsIRet(INS ins);

#endif // PIN_G_INS_API_XED_IA32_PH


