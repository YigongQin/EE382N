//Groups: @ingroup\s+(API_REF|KNOBS|IMG_BASIC_API|INS_BASIC_API|INS_INST_API|INS_BASIC_API_GEN_IA32|INS_BASIC_API_IA32|INS_MOD_API_GEN_IA32|SEC_BASIC_API|RTN_BASIC_API|REG_BASIC_API|REG_CPU_GENERIC|REG_CPU_IA32|TRACE_BASIC_API|BBL_BASIC_API|SYM_BASIC_API|MISC_PRINT|MISC_PARSE|KNOB_API|KNOB_BASIC|KNOB_PRINT|LOCK|PIN_CONTROL|TRACE_VERSION_API|BUFFER_API|PROTO_API|PIN_PROCESS_API|PIN_THREAD_API|PIN_SYSCALL_API|WINDOWS_SYSCALL_API_UNDOC|DEBUG_API|ERROR_FILE_BASIC|TYPE_BASE|INSTLIB|ALARM|CHILD_PROCESS_API|UTILS|MISC|CONTEXT_API|PHYSICAL_CONTEXT_API|PIN_CALLBACKS|EXCEPTION_API|APPDEBUG_API|STOPPED_THREAD_API|BUFFER_API|PROTO|INST_ARGS|DEPRECATED_PIN_API|INTERNAL_EXCEPTION_PRIVATE_UNDOCUMENTED|PIN_THREAD_PRIVATE|CHILD_PROCESS_INTERNAL|BBL_BASIC|ROGUE_BASIC_API|MESSAGE_TYPE|MESSAGE_BASIC|ERRFILE|MISC_BASIC|ITC_INST_API|CONTEXT_API_UNDOC|EXCEPTION_API_UNDOC|UNDOCUMENTED_PIN_API|OPIN|TRACE_VERSIONS

#ifdef PIN_G_EXCEPTION_PH
#error duplicate inclusion of exception
#else
#define PIN_G_EXCEPTION_PH
enum EXCEPTION_CLASS
{
    EXCEPTCLASS_NONE,                   ///< Reserved. No real exception belongs to this class.
    EXCEPTCLASS_UNKNOWN,                ///< Unknown exception.
    EXCEPTCLASS_ACCESS_FAULT,           ///< Memory access fault.
    EXCEPTCLASS_INVALID_INS,            ///< Invalid instruction.
    EXCEPTCLASS_INT_ERROR,              ///< Erroneous integer operation.
    EXCEPTCLASS_FP_ERROR,               ///< Erroneous floating point operation.
    EXCEPTCLASS_MULTIPLE_FP_ERROR,      ///< Received an ambiguous floating point exception.
    EXCEPTCLASS_DEBUG,                  ///< Debugging trap.
    EXCEPTCLASS_OS                      ///< O/S specific exception.
};

enum EXCEPTION_CODE
{
    EXCEPTCODE_NONE,                    ///< Reserved. No real exception has this code.

    // These exception codes are supported on all operating systems.  Portable tools should
    // use these codes when creating exceptions that wil be passed to PIN_RaiseException().

    EXCEPTCODE_ACCESS_INVALID_ADDRESS,  ///< Virtual address not mapped (the \#PF exception).
                                        ///< Belongs to EXCEPTCLASS_ACCESS_FAULT.
    EXCEPTCODE_ACCESS_DENIED,           ///< Access not permitted due to protection violation (the \#PF exception).
                                        ///< Belongs to EXCEPTCLASS_ACCESS_FAULT.
    EXCEPTCODE_ACCESS_INVALID_PAGE,     ///< A \#PF exception for some reason other than the cases above.
                                        ///< Belongs to EXCEPTCLASS_ACCESS_FAULT.
    EXCEPTCODE_ACCESS_MISALIGNED,       ///< Misaligned memory reference (the \#AC exception).
                                        ///< Belongs to EXCEPTCLASS_ACCESS_FAULT.
    EXCEPTCODE_ILLEGAL_INS,             ///< Illegal instruction (the \#UD exception).
                                        ///< Belongs to EXCEPTCLASS_INVALID_INS.
    EXCEPTCODE_PRIVILEGED_INS,          ///< Privileged instruction (the \#GP exception).
                                        ///< Belongs to EXCEPTCLASS_INVALID_INS.
    EXCEPTCODE_INT_DIVIDE_BY_ZERO,      ///< Integer divide by zero (the \#DE exception).
                                        ///< Belongs to EXCEPTCLASS_INT_ERROR.
    EXCEPTCODE_INT_OVERFLOW_TRAP,       ///< Integer overflow trap (the \#OF trap).
                                        ///< Belongs to EXCEPTCLASS_INT_ERROR.
    EXCEPTCODE_INT_BOUNDS_EXCEEDED,     ///< Array index is out of bounds (\#BR exception).
                                        ///< Belongs to EXCEPTCLASS_INT_ERROR.
    EXCEPTCODE_X87_DIVIDE_BY_ZERO,      ///< x87 FPU divide by zero (the \#MF/\#Z exception).
                                        ///< Belongs to EXCEPTCLASS_FP_ERROR.
    EXCEPTCODE_X87_OVERFLOW,            ///< x87 FPU overflow (the \#MF/\#O exception).
                                        ///< Belongs to EXCEPTCLASS_FP_ERROR.
    EXCEPTCODE_X87_UNDERFLOW,           ///< x87 FPU underflow (the \#MF/\#U exception).
                                        ///< Belongs to EXCEPTCLASS_FP_ERROR.
    EXCEPTCODE_X87_INEXACT_RESULT,      ///< x87 FPU inexact result (the \#MF/\#P exception).
                                        ///< Belongs to EXCEPTCLASS_FP_ERROR.
    EXCEPTCODE_X87_INVALID_OPERATION,   ///< x87 FPU invalid operation (the \#MF/\#IA exception).
                                        ///< Belongs to EXCEPTCLASS_FP_ERROR.
    EXCEPTCODE_X87_DENORMAL_OPERAND,    ///< x87 FPU denormal operand (the \#MF/\#D exception).
                                        ///< Belongs to EXCEPTCLASS_FP_ERROR.
    EXCEPTCODE_X87_STACK_ERROR,         ///< x87 FPU stack overflowed or underflowed (the \#MF/\#IS exception).
                                        ///< Belongs to EXCEPTCLASS_FP_ERROR.
    EXCEPTCODE_SIMD_DIVIDE_BY_ZERO,     ///< SIMD floating point divide by zero (the \#XM/\#Z exception).
                                        ///< Belongs to EXCEPTCLASS_FP_ERROR.
    EXCEPTCODE_SIMD_OVERFLOW,           ///< SIMD floating point overflow (the \#XM/\#O exception).
                                        ///< Belongs to EXCEPTCLASS_FP_ERROR.
    EXCEPTCODE_SIMD_UNDERFLOW,          ///< SIMD floating point underflow (the \#XM/\#U exception).
                                        ///< Belongs to EXCEPTCLASS_FP_ERROR.
    EXCEPTCODE_SIMD_INEXACT_RESULT,     ///< SIMD floating point inexact result (the \#XM/\#P exception).
                                        ///< Belongs to EXCEPTCLASS_FP_ERROR.
    EXCEPTCODE_SIMD_INVALID_OPERATION,  ///< SIMD invalid floating point operation (the \#XM/\#I exception).
                                        ///< Belongs to EXCEPTCLASS_FP_ERROR.
    EXCEPTCODE_SIMD_DENORMAL_OPERAND,   ///< SIMD denormal floating point operand (the \#XM/\#D exception).
                                        ///< Belongs to EXCEPTCLASS_FP_ERROR.
    EXCEPTCODE_DBG_BREAKPOINT_TRAP,     ///< Breakpoint trap (the \#BP trap).
                                        ///< Belongs to EXCEPTCLASS_DEBUG.
    EXCEPTCODE_DBG_SINGLE_STEP_TRAP,    ///< Trace trap (the \#DB trap).
                                        ///< Belongs to EXCEPTCLASS_DEBUG.

    // These exception codes can be used to raise Windows-specific exceptions.

    EXCEPTCODE_ACCESS_WINDOWS_GUARD_PAGE,       ///< Guard page access
                                                ///< (the Windows STATUS_GUARD_PAGE_VIOLATION exception).
                                                ///< Belongs to EXCEPTCLASS_ACCESS_FAULT.
    EXCEPTCODE_ACCESS_WINDOWS_STACK_OVERFLOW,   ///< Thread stack overflowed
                                                ///< (the Windows STATUS_STACK_OVERFLOW exception).
                                                ///< Belongs to EXCEPTCLASS_ACCESS_FAULT.
    EXCEPTCODE_WINDOWS,                         ///< Generic Windows exception.
                                                ///< Belongs to EXCEPTCLASS_OS.

    // When Pin receives an exception, it may report the exception using one of the codes above or
    // one of the codes in the group below.

    EXCEPTCODE_RECEIVED_UNKNOWN,                    ///< Unknown exception, cannot be re-raised.
                                                    ///< Belongs to EXCEPTCLASS_UNKNOWN.
    EXCEPTCODE_RECEIVED_ACCESS_FAULT,               ///< General memory access fault. 
                                                    ///< Belongs to EXCEPTCLASS_ACCESS_FAULT.
    EXCEPTCODE_RECEIVED_AMBIGUOUS_X87,              ///< Ambiguous x87 FPU exception.
                                                    ///< PIN_GetFpErrorSet() tell which are possible.
                                                    ///< Belongs to EXCEPTCLASS_MULTIPLE_FP_ERROR.
    EXCEPTCODE_RECEIVED_AMBIGUOUS_SIMD              ///< Ambiguous SIMD FPU exception.
                                                    ///< PIN_GetFpErrorSet() tell which are possible.
                                                    ///< Belongs to EXCEPTCLASS_MULTIPLE_FP_ERROR.
};

enum FAULTY_ACCESS_TYPE
{
    FAULTY_ACCESS_TYPE_UNKNOWN, ///< Unknown access violation
    FAULTY_ACCESS_READ,         ///< Read access
    FAULTY_ACCESS_WRITE,        ///< Write access
    FAULTY_ACCESS_EXECUTE       ///< Execute access
};

enum FPERROR
{
    FPERROR_DIVIDE_BY_ZERO =    (1<<0),     ///< The #DE exception.
    FPERROR_OVERFLOW =          (1<<1),     ///< The #O exception.
    FPERROR_UNDERFLOW =         (1<<2),     ///< The #U exception.
    FPERROR_INEXACT_RESULT =    (1<<3),     ///< The #P exception.
    FPERROR_INVALID_OPERATION = (1<<4),     ///< The #IA exception on x87 or #I exception on SIMD.
    FPERROR_DENORMAL_OPERAND =  (1<<5),     ///< The #D exception.
    FPERROR_X87_STACK_ERROR =   (1<<6)      ///< The #IS exception on x87.
};

const UINT32 MAX_WINDOWS_EXCEPTION_ARGS = 5;

/// @cond INTERNAL_DOXYGEN
struct EXCEPTION_INFO
{
    /* ================================================================================== */
    // Generic exception information
    /* ================================================================================== */

    EXCEPTION_INFO & Init(EXCEPTION_CODE exceptCode, ADDRINT exceptAddress);

    EXCEPTION_CODE GetExceptCode() const {return m_exceptCode;}

    static EXCEPTION_CLASS GetExceptClass(EXCEPTION_CODE exceptCode);
    EXCEPTION_CLASS GetExceptClass() const {return GetExceptClass(m_exceptCode);}

    ADDRINT GetExceptAddress() const {return m_exceptAddress;}

    VOID SetExceptAddress(ADDRINT exceptAddress) {m_exceptAddress = exceptAddress;}

    static BOOL IsTrap(EXCEPTION_CODE exceptCode);
    BOOL IsTrap() const {return IsTrap(m_exceptCode);}

    VOID Reset() {m_exceptCode = EXCEPTCODE_NONE; m_exceptAddress = 0; }

    BOOL IsEmpty() const {return (m_exceptCode == EXCEPTCODE_NONE);}

    std::string ToString() const;

    std::string GetCodeAsString() const;

  public:

    /* ================================================================================== */
    // EXCEPTCLASS_ACCESS_FAULT information
    /* ================================================================================== */

    EXCEPTION_INFO & InitAccessFault(EXCEPTION_CODE exceptCode,
                                     ADDRINT exceptAddress,
                                     ADDRINT accessAddress,
                                     FAULTY_ACCESS_TYPE accessType = FAULTY_ACCESS_TYPE_UNKNOWN);

    BOOL IsAccessFault() const {return GetExceptClass() == EXCEPTCLASS_ACCESS_FAULT;}

    FAULTY_ACCESS_TYPE GetFaultyAccessType() const 
    {
        ASSERTX(IsAccessFault()); 
        return m_specific.m_access.m_type;
    }

    BOOL GetFaultyAccessAddress(ADDRINT * pAccessAddress) const 
    {
        ASSERTX(IsAccessFault());
        return m_specific.m_access.m_address.Get(pAccessAddress);
    }

  public:

    /* ================================================================================== */
    // EXCEPTCLASS_MULTIPLE_FP_ERROR information
    /* ================================================================================== */

    EXCEPTION_INFO & InitMultipleFpError(EXCEPTION_CODE exceptCode,
                                         ADDRINT exceptAddress,
                                         UINT32 fpErrors);

    BOOL IsMultipleFpException() const {return GetExceptClass() == EXCEPTCLASS_MULTIPLE_FP_ERROR;}

    UINT32 GetFpErrors() const
    {
        ASSERTX(IsMultipleFpException()); 
        return m_specific.m_multipleFp.m_fpErrors;
    }

  public:

    /* ================================================================================== */
    // EXCEPTCODE_WINDOWS information
    /* ================================================================================== */

    EXCEPTION_INFO & InitWindowsSysException(UINT32 sysExceptCode,
                                          ADDRINT exceptAddress,
                                          UINT32 numArgs = 0,
                                          const ADDRINT * pArgs = 0);

    BOOL IsWindowsSysException() const {return m_exceptCode == EXCEPTCODE_WINDOWS;}

    UINT32 GetWindowsSysExceptionCode() const 
    {
        ASSERTX(IsWindowsSysException());
        return m_specific.m_winsys.m_exceptCode;
    }

    UINT32 CountWindowsSysArguments() const 
    {
        ASSERTX(IsWindowsSysException());
        return m_specific.m_winsys.m_numArgs;
    }

    ADDRINT GetWindowsSysArgument(UINT32 argNum) const 
    {
        ASSERTX(IsWindowsSysException());
        ASSERTX(argNum < m_specific.m_winsys.m_numArgs);
        return m_specific.m_winsys.m_args[argNum];
    }

  public:

    /* ================================================================================== */
    // Data
    /* ================================================================================== */
    EXCEPTION_CODE m_exceptCode;                    ///< Exception code
    ADDRINT m_exceptAddress;                        ///< Address of the faulty instruction

    union EXCEPTION_SPECIFIC
    {
        struct ACCESS_FAULT                             ///< EXCEPTCLASS_ACCESS_FAULT additional information
        {
            FAULTY_ACCESS_TYPE m_type;                  ///< The type of the faulty memory access
            OPTIONAL_VALUE<ADDRINT> m_address;          ///< Address of the faulty memory access (optional)
        }m_access;

        struct WINDOWS_SYS                              ///< EXCEPTCODE_WINDOWS additional information
        {
            UINT32 m_exceptCode;                        ///< The system exception code or signal number
            UINT32 m_numArgs;                           ///< Number of arguments in the [m_args] array
            ADDRINT m_args[MAX_WINDOWS_EXCEPTION_ARGS]; ///< Array of arguments associated with the exception
        }m_winsys;

        struct MULTIPLE_FP                              ///< EXCEPTCLASS_MULTIPLE_FP_ERROR additional information
        {
            UINT32 m_fpErrors;                          ///< Bit mask of FPERROR values
        }m_multipleFp;
    }m_specific;
};

/// @endcond
typedef struct EXCEPTION_INFO EXCEPTION_INFO;

inline VOID PIN_InitExceptionInfo(EXCEPTION_INFO * pExceptInfo,
                                        EXCEPTION_CODE exceptCode,
                                        ADDRINT exceptAddress)
{
    pExceptInfo->Init(exceptCode, exceptAddress);
}

inline VOID PIN_InitAccessFaultInfo(EXCEPTION_INFO * pExceptInfo,
                                          EXCEPTION_CODE exceptCode,
                                          ADDRINT exceptAddress,
                                          ADDRINT accessAddress,
                                          FAULTY_ACCESS_TYPE accessType = FAULTY_ACCESS_TYPE_UNKNOWN)
{
    pExceptInfo->InitAccessFault(exceptCode, exceptAddress, accessAddress, accessType);
}

inline VOID PIN_InitWindowsExceptionInfo(EXCEPTION_INFO * pExceptInfo,
                                               UINT32 sysExceptCode,
                                               ADDRINT exceptAddress,
                                               UINT32 numArgs = 0,
                                               const ADDRINT * pArgs = 0)
{
    pExceptInfo->InitWindowsSysException(sysExceptCode, exceptAddress, numArgs, pArgs);
}

inline EXCEPTION_CODE PIN_GetExceptionCode(const EXCEPTION_INFO * pExceptInfo)
{
    return pExceptInfo->GetExceptCode();
}

inline EXCEPTION_CLASS PIN_GetExceptionClass(EXCEPTION_CODE exceptCode)
{
    return EXCEPTION_INFO::GetExceptClass(exceptCode);
}

inline ADDRINT PIN_GetExceptionAddress(const EXCEPTION_INFO * pExceptInfo)
{
    return pExceptInfo->GetExceptAddress();
}

inline VOID PIN_SetExceptionAddress(EXCEPTION_INFO * pExceptInfo,
                                          ADDRINT exceptAddress)
{
    pExceptInfo->SetExceptAddress(exceptAddress);
}

inline FAULTY_ACCESS_TYPE PIN_GetFaultyAccessType(const EXCEPTION_INFO * pExceptInfo)
{
    return pExceptInfo->GetFaultyAccessType();
}

inline BOOL PIN_GetFaultyAccessAddress(const EXCEPTION_INFO * pExceptInfo,
                                             ADDRINT * pAccessAddress)
{
    return pExceptInfo->GetFaultyAccessAddress(pAccessAddress);
}

inline UINT32 PIN_GetFpErrorSet(const EXCEPTION_INFO * pExceptInfo)
{
    return pExceptInfo->GetFpErrors();
}

inline UINT32 PIN_GetWindowsExceptionCode(const EXCEPTION_INFO * pExceptInfo)
{
    return pExceptInfo->GetWindowsSysExceptionCode();
}

inline UINT32 PIN_CountWindowsExceptionArguments(const EXCEPTION_INFO * pExceptInfo)
{
    return pExceptInfo->CountWindowsSysArguments();
}

inline ADDRINT PIN_GetWindowsExceptionArgument(const EXCEPTION_INFO * pExceptInfo,
                                                     UINT32 argNum)
{
    return pExceptInfo->GetWindowsSysArgument(argNum);
}

inline std::string PIN_ExceptionToString(const EXCEPTION_INFO * pExceptInfo)
{
    return pExceptInfo->ToString();
}

#endif // PIN_G_EXCEPTION_PH


