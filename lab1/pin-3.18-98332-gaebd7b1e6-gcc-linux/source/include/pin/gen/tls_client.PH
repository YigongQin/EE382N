//Groups: @ingroup\s+(API_REF|KNOBS|IMG_BASIC_API|INS_BASIC_API|INS_INST_API|INS_BASIC_API_GEN_IA32|INS_BASIC_API_IA32|INS_MOD_API_GEN_IA32|SEC_BASIC_API|RTN_BASIC_API|REG_BASIC_API|REG_CPU_GENERIC|REG_CPU_IA32|TRACE_BASIC_API|BBL_BASIC_API|SYM_BASIC_API|MISC_PRINT|MISC_PARSE|KNOB_API|KNOB_BASIC|KNOB_PRINT|LOCK|PIN_CONTROL|TRACE_VERSION_API|BUFFER_API|PROTO_API|PIN_PROCESS_API|PIN_THREAD_API|PIN_SYSCALL_API|WINDOWS_SYSCALL_API_UNDOC|DEBUG_API|ERROR_FILE_BASIC|TYPE_BASE|INSTLIB|ALARM|CHILD_PROCESS_API|UTILS|MISC|CONTEXT_API|PHYSICAL_CONTEXT_API|PIN_CALLBACKS|EXCEPTION_API|APPDEBUG_API|STOPPED_THREAD_API|BUFFER_API|PROTO|INST_ARGS|DEPRECATED_PIN_API|INTERNAL_EXCEPTION_PRIVATE_UNDOCUMENTED|PIN_THREAD_PRIVATE|CHILD_PROCESS_INTERNAL|BBL_BASIC|ROGUE_BASIC_API|MESSAGE_TYPE|MESSAGE_BASIC|ERRFILE|MISC_BASIC|ITC_INST_API|CONTEXT_API_UNDOC|EXCEPTION_API_UNDOC|UNDOCUMENTED_PIN_API|OPIN|TRACE_VERSIONS

#ifdef PIN_G_TLS_CLIENT_PH
#error duplicate inclusion of tls_client
#else
#define PIN_G_TLS_CLIENT_PH
/// @cond INTERNAL_DOXYGEN
enum
{
    // Reserved for internal exception handling
    TLS_KEY_INTERNAL_EXCEPTION,

    TLS_KEY_CLIENT_FIRST,
    // Keys in the [TLS_KEY_CLIENT_FIRST, TLS_KEY_CLIENT_LAST] range can be
    // allocated by tool
    TLS_KEY_CLIENT_LAST = TLS_KEY_CLIENT_FIRST + 63
};

class CLIENT_TLS
{
  public:

    TLS_KEY AllocateKey(DESTRUCTFUN destructFun = NULL);

    BOOL FreeKey(TLS_KEY tlsKey);

    BOOL IsKeyAllocated(TLS_KEY tlsKey);

    static VOID * GetData(TLS_KEY tlsKey, THREADID threadId)
    {
        return (*(m_pSlots[threadId]))[tlsKey];
    }

    static BOOL SetData(TLS_KEY tlsKey, const VOID *  value, THREADID threadId)
    {
        (*(m_pSlots[threadId]))[tlsKey] = const_cast<VOID *>(value);
        return TRUE;
    }

    VOID InitCurrentTls(THREADID currentThreadId);

    VOID DestructThreadTls(THREADID threadId);

    BOOL IsCurrentTlsAllocated(THREADID currentThreadId);

    static BOOL IsValidTlsKey(TLS_KEY tlsKey, THREADID threadId)
    {
        return m_pSlots[threadId]->IsValidKey(tlsKey) && CLIENT_TLS::Instance()->IsKeyAllocated(tlsKey);
    }

    static CLIENT_TLS * Instance();

  private:
    typedef TLS_ARRAY<0, TLS_KEY_CLIENT_LAST> SLOTS;

  private:
    // Private constructor
    CLIENT_TLS() {}
    friend class DUMMY_GCC32; //dummy friend to appease gcc 3.2.X compiler. It does not understand
                              //singleton idiom and reports warning for private constructor.
    //Disable copy constructor and assignment operator
    CLIENT_TLS(const CLIENT_TLS &);
    CLIENT_TLS & operator = (const CLIENT_TLS &);

    static SLOTS * m_pSlots[PIN_MAX_THREADS];

    /*
     * Pointer to a single instance of this class
     * This static member enforces initialization of CLIENT_TLS in the first (main) thread.
     */
    static CLIENT_TLS * m_pInstance;

};

/// @endcond
const UINT32 MAX_CLIENT_TLS_KEYS = (TLS_KEY_CLIENT_LAST - TLS_KEY_CLIENT_FIRST + 1);

inline TLS_KEY PIN_CreateThreadDataKey(DESTRUCTFUN destruct_func)
{
    return CLIENT_TLS::Instance()->AllocateKey(destruct_func);
}

inline BOOL PIN_DeleteThreadDataKey(TLS_KEY key)
{
    if (CLIENT_TLS::Instance()->IsKeyAllocated(key))
        return CLIENT_TLS::Instance()->FreeKey(key);
    else
        return FALSE;
}

/* This version of PIN_SetThreadData is now deprecated. Please use the version of this function which takes
 * a THREADID argument.
 */
inline BOOL PIN_SetThreadData( TLS_KEY key, const VOID * data )
{
    ASSERT(FALSE, "This version of PIN_SetThreadData is now deprecated. Please use the version of this function which "
                  "takes a THREADID argument.");
    return FALSE;
}

inline BOOL PIN_SetThreadData( TLS_KEY key, const VOID * data , THREADID threadId)
{
    if (CLIENT_TLS::IsValidTlsKey(key, threadId))
        return CLIENT_TLS::SetData( key, data, threadId );
    else
        return FALSE;
}

/* This version of PIN_GetThreadData is now deprecated. Please use the version of this function which takes
 * a THREADID argument.
 */
inline VOID * PIN_GetThreadData( TLS_KEY key )
{
    ASSERT(FALSE, "This version of PIN_GetThreadData is now deprecated. Please use the version of this function which "
                  "takes a THREADID argument.");
    return FALSE;
}

inline VOID * PIN_GetThreadData( TLS_KEY key, THREADID threadId )
{
    if (CLIENT_TLS::IsValidTlsKey(key, threadId))
        return CLIENT_TLS::GetData( key, threadId );
    else
        return NULL;
}

#endif // PIN_G_TLS_CLIENT_PH


