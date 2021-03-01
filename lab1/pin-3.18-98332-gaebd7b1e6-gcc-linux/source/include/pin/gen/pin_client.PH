//Groups: @ingroup\s+(API_REF|KNOBS|IMG_BASIC_API|INS_BASIC_API|INS_INST_API|INS_BASIC_API_GEN_IA32|INS_BASIC_API_IA32|INS_MOD_API_GEN_IA32|SEC_BASIC_API|RTN_BASIC_API|REG_BASIC_API|REG_CPU_GENERIC|REG_CPU_IA32|TRACE_BASIC_API|BBL_BASIC_API|SYM_BASIC_API|MISC_PRINT|MISC_PARSE|KNOB_API|KNOB_BASIC|KNOB_PRINT|LOCK|PIN_CONTROL|TRACE_VERSION_API|BUFFER_API|PROTO_API|PIN_PROCESS_API|PIN_THREAD_API|PIN_SYSCALL_API|WINDOWS_SYSCALL_API_UNDOC|DEBUG_API|ERROR_FILE_BASIC|TYPE_BASE|INSTLIB|ALARM|CHILD_PROCESS_API|UTILS|MISC|CONTEXT_API|PHYSICAL_CONTEXT_API|PIN_CALLBACKS|EXCEPTION_API|APPDEBUG_API|STOPPED_THREAD_API|BUFFER_API|PROTO|INST_ARGS|DEPRECATED_PIN_API|INTERNAL_EXCEPTION_PRIVATE_UNDOCUMENTED|PIN_THREAD_PRIVATE|CHILD_PROCESS_INTERNAL|BBL_BASIC|ROGUE_BASIC_API|MESSAGE_TYPE|MESSAGE_BASIC|ERRFILE|MISC_BASIC|ITC_INST_API|CONTEXT_API_UNDOC|EXCEPTION_API_UNDOC|UNDOCUMENTED_PIN_API|OPIN|TRACE_VERSIONS

#ifdef PIN_G_PIN_CLIENT_PH
#error duplicate inclusion of pin_client
#else
#define PIN_G_PIN_CLIENT_PH

#define EnterPinClientSlaveMode() CheckPinClientLock(__FUNCTION__)
#define ExitPinClientSlaveMode() CheckPinClientLock(__FUNCTION__)
#ifndef PIN_GENERATED_MESSAGE_PIN_H
#define LOG_IMAGE(s)         if(ClientInt()->_VmLog) ClientInt()->_VmLog("IMAGE", s);
#define LOG_SYM(s)           if(ClientInt()->_VmLog) ClientInt()->_VmLog("SYMBOLS", s);
#define LOG_PROBE(s)         if(ClientInt()->_VmLog) ClientInt()->_VmLog("PROBE", s);
#define LOG_FETCH_RTN_INS(s) if(ClientInt()->_VmLog) ClientInt()->_VmLog("FETCH_RTN_INS", s);
#define LOG_UNWIND(s)        if(ClientInt()->_VmLog) ClientInt()->_VmLog("UNWIND", s);
#define LOG_BUFFER(s)        if(ClientInt()->_VmLog) ClientInt()->_VmLog("BUFFER", s);
#define LOG_JIT_PROFILING(s) if(ClientInt()->_VmLog) ClientInt()->_VmLog("VTUNE_JIT_API", s);
// When LOG()/WARNING() is called from Pin client library it will be printed to pintool.log
// If one wants to print a general log/warning message into pin.log, 
// then LOG_VMLOG()/LOG_VMWARNING() macro should be used.
#define LOG_VMLOG(s)          if(ClientInt()->_VmLog) ClientInt()->_VmLog("LOG", s);
#define LOG_VMWARNING(s)      if(ClientInt()->_VmLog) ClientInt()->_VmLog("WARNING", s);
#endif
class TRACE_CLASS;

typedef TRACE_CLASS * TRACE;

typedef VOID * PIN_CONFIGURATION_INFO;

class COMPLEX_CALLBACKVAL_BASE;

typedef COMPLEX_CALLBACKVAL_BASE * PIN_CALLBACK;

const PIN_CALLBACK PIN_CALLBACK_INVALID(0);

typedef VOID (*REMOVE_INSTRUMENTATION_CALLBACK)(VOID *v);

typedef VOID (*DETACH_CALLBACK)(VOID *v);

typedef VOID (*DETACH_PROBED_CALLBACK)(VOID *v);

typedef VOID (*ATTACH_PROBED_CALLBACK)(VOID *v);

typedef VOID (*ATTACH_CALLBACK)(VOID *v);

typedef VOID (*APPLICATION_START_CALLBACK)(VOID *v);

typedef VOID (*PREPARE_FOR_FINI_CALLBACK)(VOID *v);

typedef VOID (*FINI_CALLBACK)(INT32 code, VOID *v);

typedef size_t (*FETCH_CALLBACK)(void *buf, ADDRINT addr, size_t size,
    EXCEPTION_INFO *pExceptInfo, VOID *v);

typedef VOID (*OUT_OF_MEMORY_CALLBACK)(size_t size, VOID *v);

typedef ADDRINT (PIN_FAST_ANALYSIS_CALL * MEMORY_ADDR_TRANS_CALLBACK)(PIN_MEM_TRANS_INFO* memTransInfo, VOID *v);

typedef VOID (*SMC_CALLBACK)(ADDRINT traceStartAddress, ADDRINT traceEndAddress, VOID *v);

typedef VOID (*FORK_CALLBACK)(THREADID threadid, const CONTEXT *ctxt, VOID *v);

typedef VOID (*THREAD_START_CALLBACK)(THREADID threadIndex, CONTEXT *ctxt, INT32 flags, VOID *v);

typedef VOID (*THREAD_ATTACH_PROBED_CALLBACK)(VOID *sigmask, VOID *v);

typedef VOID (*THREAD_DETACH_PROBED_CALLBACK)(VOID *v);

typedef VOID (*THREAD_ATTACH_CALLBACK)(THREADID threadIndex, CONTEXT *ctxt, VOID *v);

typedef VOID (*THREAD_DETACH_CALLBACK)(THREADID threadIndex, const CONTEXT *ctxt, VOID *v);

typedef VOID (*THREAD_FINI_CALLBACK)(THREADID threadIndex, const CONTEXT *ctxt, INT32 code, VOID *v);

typedef VOID (*CONTEXT_CHANGE_CALLBACK)(THREADID threadIndex, CONTEXT_CHANGE_REASON reason,
                                                   const CONTEXT *from, CONTEXT *to, INT32 info, VOID *v);

typedef VOID (*SYSCALL_ENTRY_CALLBACK)(THREADID threadIndex, CONTEXT *ctxt,
                                                  SYSCALL_STANDARD std, VOID *v);

typedef VOID (*SYSCALL_EXIT_CALLBACK)(THREADID threadIndex, CONTEXT *ctxt,
                                                 SYSCALL_STANDARD std, VOID *v);

typedef BOOL (*INTERCEPT_SIGNAL_CALLBACK)(THREADID tid, INT32 sig, CONTEXT *ctxt,
    BOOL hasHandler, const EXCEPTION_INFO *pExceptInfo, VOID *v);

typedef VOID * (*TRACE_BUFFER_CALLBACK)(BUFFER_ID id, THREADID tid, const CONTEXT *ctxt,
                           VOID *buf, UINT64 numElements, VOID *v);

typedef VOID (*INS_INSTRUMENT_CALLBACK)(INS ins, VOID *v);

typedef VOID (*TRACE_INSTRUMENT_CALLBACK)(TRACE trace, VOID *v);

typedef VOID (*RTN_INSTRUMENT_CALLBACK)(RTN rtn, VOID *v);

/// @cond INTERNAL_DOXYGEN
enum REPLAY_MODE
{
    REPLAY_MODE_NONE            = 0x00,
    REPLAY_MODE_IMAGEOPS        = 0x01,   ///< Replayer will handle image ops
    REPLAY_MODE_ALL             = REPLAY_MODE_IMAGEOPS
};

extern MESSAGE_TYPE MessageTypeLogProbe;

extern MESSAGE_TYPE MessageTypeLogBuffer;

extern MESSAGE_TYPE MessageTypeLogUnwind;

extern MESSAGE_TYPE MessageTypeLogFetchRtnIns;

extern VOID PIN_SetReplayMode(UINT32 flags);

extern BOOL PIN_SpawnApplicationThread(const CONTEXT * ctxt);

/// @endcond
extern VOID PIN_LockClient();

extern VOID PIN_UnlockClient();

extern VOID PIN_WriteErrorMessage( const char * msg, INT32 type, PIN_ERR_SEVERITY_TYPE severity, INT32 num, ... );

extern AFUNPTR RTN_Replace(RTN replacedRtn, AFUNPTR replacementFun);

extern AFUNPTR RTN_ReplaceSignature(RTN replacedRtn, AFUNPTR replacementFun, ...);

enum PROBE_MODE
{
    PROBE_MODE_DEFAULT = 0,
    PROBE_MODE_ALLOW_RELOCATION = (1<<0)
};

extern AFUNPTR RTN_ReplaceProbed(RTN replacedRtn, AFUNPTR replacementFun);

extern AFUNPTR RTN_ReplaceProbedEx(RTN replacedRtn, PROBE_MODE mode,
                                 AFUNPTR replacementFun);

extern VOID PIN_CallApplicationFunction(const CONTEXT * ctxt, THREADID tid, CALLINGSTD_TYPE cstype,
    AFUNPTR origFunPtr, CALL_APPLICATION_FUNCTION_PARAM* param, ... );

extern BOOL INS_IsAddedForFunctionReplacement(INS ins);

extern VOID CALLBACK_SetExecutionOrder(PIN_CALLBACK callback, CALL_ORDER order);

extern CALL_ORDER CALLBACK_GetExecutionOrder(PIN_CALLBACK callback);

extern PIN_CALLBACK TRACE_AddInstrumentFunction(TRACE_INSTRUMENT_CALLBACK fun, VOID *val);

extern PIN_CALLBACK INS_AddInstrumentFunction(INS_INSTRUMENT_CALLBACK fun, VOID *val);

extern PIN_CALLBACK PIN_AddFiniFunction(FINI_CALLBACK fun, VOID *val);

extern PIN_CALLBACK PIN_AddPrepareForFiniFunction(PREPARE_FOR_FINI_CALLBACK fun, VOID *val);

extern VOID PIN_AddFetchFunction(FETCH_CALLBACK fun, VOID *val);

extern size_t PIN_FetchCode(void *copyBuf, const VOID *address, size_t maxSize,
                               EXCEPTION_INFO *pExceptInfo);

extern VOID PIN_AddOutOfMemoryFunction(OUT_OF_MEMORY_CALLBACK fun, VOID *val);

extern VOID PIN_AddMemoryAddressTransFunction(MEMORY_ADDR_TRANS_CALLBACK fun, VOID *val);

extern VOID TRACE_AddSmcDetectedFunction(SMC_CALLBACK fun, VOID *val);

extern MEMORY_ADDR_TRANS_CALLBACK PIN_GetMemoryAddressTransFunction();

extern PIN_CALLBACK PIN_AddDetachFunction(DETACH_CALLBACK fun, VOID *val);

extern PIN_CALLBACK PIN_AddDetachFunctionProbed(DETACH_PROBED_CALLBACK fun, VOID *val);

extern PIN_CALLBACK PIN_AddThreadStartFunction(THREAD_START_CALLBACK fun, VOID *val);

extern PIN_CALLBACK PIN_AddThreadAttachProbedFunction(THREAD_ATTACH_PROBED_CALLBACK fun, VOID *val);

extern PIN_CALLBACK PIN_AddThreadAttachFunction(THREAD_ATTACH_CALLBACK fun, VOID *val);

extern PIN_CALLBACK PIN_AddThreadDetachProbedFunction(
        THREAD_DETACH_PROBED_CALLBACK fun, VOID *val);

extern PIN_CALLBACK PIN_AddThreadDetachFunction(THREAD_DETACH_CALLBACK fun, VOID *val);

extern PIN_CALLBACK PIN_AddApplicationStartFunction(APPLICATION_START_CALLBACK fun, VOID *val);

extern PIN_CALLBACK PIN_AddThreadFiniFunction(THREAD_FINI_CALLBACK fun, VOID *val);

extern PIN_CALLBACK PIN_AddContextChangeFunction(CONTEXT_CHANGE_CALLBACK fun, VOID *val);

extern PIN_CALLBACK PIN_AddSyscallEntryFunction(SYSCALL_ENTRY_CALLBACK fun, VOID *val);

extern PIN_CALLBACK PIN_AddSyscallExitFunction(SYSCALL_EXIT_CALLBACK fun, VOID *val);

extern BOOL PIN_InterceptSignal(INT32 sig, INTERCEPT_SIGNAL_CALLBACK fun, VOID *val);

extern BOOL PIN_UnblockSignal(INT32 sig, BOOL enable);

typedef enum
{
    SMC_ENABLE,
    SMC_DISABLE
} SMC_ENABLE_DISABLE_TYPE;

extern VOID PIN_SetSmcSupport(SMC_ENABLE_DISABLE_TYPE enable_disable);

enum FPOINT
{
    FPOINT_BEFORE,          ///< Call-back in parent, just before fork
    FPOINT_AFTER_IN_PARENT, ///< Call-back in parent, immediately after fork
    FPOINT_AFTER_IN_CHILD   ///< Call-back in child, immediately after fork
};

extern PIN_CALLBACK PIN_AddForkFunction(FPOINT point, FORK_CALLBACK fun, VOID *val);

extern VOID PIN_RemoveInstrumentation();

extern VOID PIN_RemoveInstrumentationInRange(ADDRINT start, ADDRINT end);

extern VOID PIN_RemoveFiniFunctions();

extern VOID PIN_Detach();

enum ATTACH_STATUS
{
    ATTACH_INITIATED,       ///< Attach request was delivered successfully to PIN
    ATTACH_FAILED_DETACH    ///< Attach request was not delivered to Pin
                            ///< since detach session hasn't completed yet.
};

extern ATTACH_STATUS PIN_Attach(ATTACH_CALLBACK fun, VOID *val);

extern VOID PIN_DetachProbed();

extern ATTACH_STATUS PIN_AttachProbed(ATTACH_PROBED_CALLBACK fun, VOID *val);

extern VOID INS_InsertPredicatedCall(INS ins, IPOINT ipoint, AFUNPTR funptr, ...);

extern VOID INS_InsertCall(INS ins, IPOINT action, AFUNPTR funptr, ...);

extern VOID INS_InsertIfCall(INS ins, IPOINT action, AFUNPTR funptr, ...);

extern VOID INS_InsertThenCall(INS ins, IPOINT action, AFUNPTR funptr, ...);

extern VOID INS_InsertIfPredicatedCall(INS ins, IPOINT action, AFUNPTR funptr, ...);

extern VOID INS_InsertThenPredicatedCall(INS ins, IPOINT action, AFUNPTR funptr, ...);

extern PIN_CONFIGURATION_INFO PIN_CreateDefaultConfigurationInfo();

extern VOID PIN_SetAllowedConcurrentCallbacks(PIN_CONFIGURATION_INFO options, PIN_CALLBACK_TYPE callbackType);

extern VOID PIN_StartProgram(PIN_CONFIGURATION_INFO options=PIN_CreateDefaultConfigurationInfo());

extern VOID PIN_StartProgram(PIN_CONFIGURATION_INFO options);

extern VOID PIN_StartProgramProbed();

extern const std::string& PIN_Version();

extern BOOL PIN_Init(INT32 argc, CHAR ** argv);

extern VOID TRACE_InsertCall(TRACE trace, IPOINT action, AFUNPTR funptr, ...);

extern VOID TRACE_InsertIfCall(TRACE trace, IPOINT action, AFUNPTR funptr, ...);

extern VOID TRACE_InsertThenCall(TRACE trace, IPOINT action, AFUNPTR funptr, ...);

extern BBL TRACE_BblHead(TRACE trace);

extern BBL TRACE_BblTail(TRACE trace);

extern ADDRINT TRACE_Address(TRACE trace);

extern USIZE TRACE_Size(TRACE trace);

extern RTN TRACE_Rtn(TRACE trace);

extern BOOL TRACE_HasFallThrough(TRACE trace);

extern UINT32 TRACE_NumBbl(TRACE trace);

extern UINT32 TRACE_NumIns(TRACE trace);

extern VOID BBL_InsertCall(BBL bbl, IPOINT action, AFUNPTR funptr, ...);

extern VOID BBL_InsertIfCall(BBL bbl, IPOINT action, AFUNPTR funptr, ...);

extern VOID BBL_InsertThenCall(BBL bbl, IPOINT action, AFUNPTR funptr, ...);

extern BOOL BBL_HasFallThrough(BBL bbl);

extern BOOL PIN_SupportsProcessorState(PROCESSOR_STATE state);

extern BOOL PIN_ContextContainsState(CONTEXT * ctxt, PROCESSOR_STATE state);

extern VOID PIN_SetContextRegval(CONTEXT * ctxt, REG reg, const UINT8 * val);

extern VOID PIN_GetContextRegval(const CONTEXT * ctxt, REG reg, UINT8 * val);

extern VOID PIN_SetContextReg(CONTEXT * ctxt, REG reg, ADDRINT val);

extern ADDRINT PIN_GetContextReg(const CONTEXT * ctxt, REG reg);

extern VOID PIN_SetContextFPState(CONTEXT * ctxt, const FPSTATE * fpstate);

extern VOID PIN_GetContextFPState(const CONTEXT * ctxt, FPSTATE * fpstate);

extern REGSET PIN_GetFullContextRegsSet();

extern VOID PIN_SaveContext(const CONTEXT * ctxtFrom, CONTEXT * ctxtTo);

extern VOID PIN_ExecuteAt(const CONTEXT * ctxt);

extern VOID PIN_SetSyscallArgument(CONTEXT *ctxt, SYSCALL_STANDARD std, UINT32 argNum, ADDRINT val);

extern ADDRINT PIN_GetSyscallArgument(const CONTEXT *ctxt, SYSCALL_STANDARD std, UINT32 argNum);

extern VOID PIN_SetSyscallNumber(CONTEXT *ctxt, SYSCALL_STANDARD std, ADDRINT val);

extern ADDRINT PIN_GetSyscallNumber(const CONTEXT *ctxt, SYSCALL_STANDARD std);

extern ADDRINT PIN_GetSyscallReturn(const CONTEXT *ctxt, SYSCALL_STANDARD std);

extern ADDRINT PIN_GetSyscallErrno(const CONTEXT *ctxt, SYSCALL_STANDARD std);

extern VOID PIN_SetPhysicalContextReg(PHYSICAL_CONTEXT * pPhysCtxt, REG reg, ADDRINT val);

extern ADDRINT PIN_GetPhysicalContextReg(const PHYSICAL_CONTEXT * pPhysCtxt, REG reg);

extern VOID PIN_SetPhysicalContextFPState(PHYSICAL_CONTEXT * pPhysCtxt, const VOID * fpstate);

extern VOID PIN_GetPhysicalContextFPState(const PHYSICAL_CONTEXT * pPhysCtxt, VOID * fpstate);

extern VOID PIN_RaiseException(const CONTEXT * ctxt, THREADID tid, const EXCEPTION_INFO * pExceptInfo);

extern VOID INS_RewriteMemoryOperand(INS ins,
                                        UINT32 memindex,
                                        REG reg);

extern VOID INS_InsertIndirectJump(INS ins, IPOINT ipoint, REG reg);

extern VOID INS_InsertDirectJump(INS ins, IPOINT ipoint, ADDRINT tgt);

extern REG PIN_ClaimToolRegister();

extern CHAR * PIN_VmFullPath();

extern const CHAR * PIN_ToolFullPath();

extern NORETURN VOID PIN_ExitProcess(INT32 exitCode);

extern INT PIN_GetPid();

extern VOID INS_Delete(INS ins);

extern VOID IARGLIST_AddArguments(IARGLIST args,  ...);

extern IARGLIST IARGLIST_Alloc();

extern VOID IARGLIST_Free(IARGLIST args);

extern size_t PIN_SafeCopy(VOID * dst, const VOID * src, size_t size);

extern size_t PIN_SafeCopyEx(VOID * dst,
                                const VOID * src,
                                size_t size,
                                EXCEPTION_INFO * pExceptInfo);

extern VOID INS_InsertFillBuffer(INS ins, IPOINT action, BUFFER_ID id, ...);

extern VOID INS_InsertFillBufferPredicated(INS ins, IPOINT action, BUFFER_ID id, ...);

extern VOID INS_InsertFillBufferThen(INS ins, IPOINT action, BUFFER_ID id, ...);

extern BUFFER_ID PIN_DefineTraceBuffer(size_t recordSize, UINT32 numPages, TRACE_BUFFER_CALLBACK fun, VOID *val);

extern VOID *PIN_AllocateBuffer(BUFFER_ID id);

extern VOID PIN_DeallocateBuffer(BUFFER_ID id, VOID *buf);

extern BOOL PIN_IsActionPending(THREADID tid);

extern UINT32 PIN_GetInitialThreadCount();

extern VOID * PIN_GetBufferPointer(CONTEXT * const ctxt, BUFFER_ID id);

extern BOOL PIN_CheckReadAccess(VOID * addr);

extern BOOL PIN_CheckWriteAccess(VOID * addr);

extern ADDRINT PIN_GetAuxVectorValue(ADDRINT type, bool *found);

#endif // PIN_G_PIN_CLIENT_PH


