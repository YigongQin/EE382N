//Groups: @ingroup\s+(API_REF|KNOBS|IMG_BASIC_API|INS_BASIC_API|INS_INST_API|INS_BASIC_API_GEN_IA32|INS_BASIC_API_IA32|INS_MOD_API_GEN_IA32|SEC_BASIC_API|RTN_BASIC_API|REG_BASIC_API|REG_CPU_GENERIC|REG_CPU_IA32|TRACE_BASIC_API|BBL_BASIC_API|SYM_BASIC_API|MISC_PRINT|MISC_PARSE|KNOB_API|KNOB_BASIC|KNOB_PRINT|LOCK|PIN_CONTROL|TRACE_VERSION_API|BUFFER_API|PROTO_API|PIN_PROCESS_API|PIN_THREAD_API|PIN_SYSCALL_API|WINDOWS_SYSCALL_API_UNDOC|DEBUG_API|ERROR_FILE_BASIC|TYPE_BASE|INSTLIB|ALARM|CHILD_PROCESS_API|UTILS|MISC|CONTEXT_API|PHYSICAL_CONTEXT_API|PIN_CALLBACKS|EXCEPTION_API|APPDEBUG_API|STOPPED_THREAD_API|BUFFER_API|PROTO|INST_ARGS|DEPRECATED_PIN_API|INTERNAL_EXCEPTION_PRIVATE_UNDOCUMENTED|PIN_THREAD_PRIVATE|CHILD_PROCESS_INTERNAL|BBL_BASIC|ROGUE_BASIC_API|MESSAGE_TYPE|MESSAGE_BASIC|ERRFILE|MISC_BASIC|ITC_INST_API|CONTEXT_API_UNDOC|EXCEPTION_API_UNDOC|UNDOCUMENTED_PIN_API|OPIN|TRACE_VERSIONS

#ifdef PIN_G_CHILD_PROCESS_PH
#error duplicate inclusion of child_process
#else
#define PIN_G_CHILD_PROCESS_PH
/// @cond INTERNAL_DOXYGEN
class CHILD_PROCESS_CLASS
{
  public:
    CHILD_PROCESS_CLASS(UINT32 processId, 
                        const COMMAND_LINE_ARGUMENTS & appCmd, 
                        const COMMAND_LINE_ARGUMENTS & pinCmd) : m_processId(processId),
                                                                 m_appCmd(appCmd),
                                                                 m_pinCmd(pinCmd),
                                                                 m_isPinCmdModified(FALSE)
    {
        (m_pinAndAppCmd = m_pinCmd).Insert(m_appCmd);
    }

    UINT32 GetId()
    {
        return m_processId;
    }

    BOOL GetApplicationCommandLine(INT * pArgc, const CHAR * const ** pArgv);

    BOOL SetPinCommandLine(INT argc, const CHAR * const * argv);

    BOOL GetPinCommandLine(INT * pArgc, const CHAR * const ** pArgv);

    BOOL GetFullCommandLine(INT * pArgc, const CHAR * const ** pArgv);

    BOOL IsPinCommandLineModified();

    ~CHILD_PROCESS_CLASS() {}

  private:
    UINT32                 m_processId;        //!< Process id of child process
    COMMAND_LINE_ARGUMENTS m_appCmd;           //!< Application's command line arguments
    COMMAND_LINE_ARGUMENTS m_pinCmd;           //!< Pin's command line arguments
    COMMAND_LINE_ARGUMENTS m_pinAndAppCmd;     //!< Pin's and Application's command line arguments
    BOOL                   m_isPinCmdModified; //!< Has the user modified pin command line?
};

/// @endcond
#endif // PIN_G_CHILD_PROCESS_PH


