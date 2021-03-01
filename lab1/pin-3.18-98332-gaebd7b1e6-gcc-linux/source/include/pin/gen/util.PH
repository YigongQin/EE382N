//Groups: @ingroup\s+(API_REF|KNOBS|IMG_BASIC_API|INS_BASIC_API|INS_INST_API|INS_BASIC_API_GEN_IA32|INS_BASIC_API_IA32|INS_MOD_API_GEN_IA32|SEC_BASIC_API|RTN_BASIC_API|REG_BASIC_API|REG_CPU_GENERIC|REG_CPU_IA32|TRACE_BASIC_API|BBL_BASIC_API|SYM_BASIC_API|MISC_PRINT|MISC_PARSE|KNOB_API|KNOB_BASIC|KNOB_PRINT|LOCK|PIN_CONTROL|TRACE_VERSION_API|BUFFER_API|PROTO_API|PIN_PROCESS_API|PIN_THREAD_API|PIN_SYSCALL_API|WINDOWS_SYSCALL_API_UNDOC|DEBUG_API|ERROR_FILE_BASIC|TYPE_BASE|INSTLIB|ALARM|CHILD_PROCESS_API|UTILS|MISC|CONTEXT_API|PHYSICAL_CONTEXT_API|PIN_CALLBACKS|EXCEPTION_API|APPDEBUG_API|STOPPED_THREAD_API|BUFFER_API|PROTO|INST_ARGS|DEPRECATED_PIN_API|INTERNAL_EXCEPTION_PRIVATE_UNDOCUMENTED|PIN_THREAD_PRIVATE|CHILD_PROCESS_INTERNAL|BBL_BASIC|ROGUE_BASIC_API|MESSAGE_TYPE|MESSAGE_BASIC|ERRFILE|MISC_BASIC|ITC_INST_API|CONTEXT_API_UNDOC|EXCEPTION_API_UNDOC|UNDOCUMENTED_PIN_API|OPIN|TRACE_VERSIONS

#ifdef PIN_G_UTIL_PH
#error duplicate inclusion of util
#else
#define PIN_G_UTIL_PH
class COMMAND_LINE_ARGUMENTS
{
  public:
    COMMAND_LINE_ARGUMENTS()
    {
        Enter(0, NULL, NULL);
    }

    COMMAND_LINE_ARGUMENTS(INT argc, const CHAR * const * argv, const CHAR * delimiter = NULL)
    {
        Enter(argc, argv, delimiter);
    }

    COMMAND_LINE_ARGUMENTS(const std::string & commandLine)
    {
        Enter(commandLine);
    }

    COMMAND_LINE_ARGUMENTS(const COMMAND_LINE_ARGUMENTS & source)
    {
        Enter(source.m_argc, source.m_argv, NULL);
    }

    COMMAND_LINE_ARGUMENTS & operator= (const COMMAND_LINE_ARGUMENTS & source)
    {
        //Avoid self assignment
        if(this != &source)
        {
            Clean();
            Enter(source.m_argc, source.m_argv, NULL);
        }
        return *this;
    }

    COMMAND_LINE_ARGUMENTS & Insert (const COMMAND_LINE_ARGUMENTS & right, INT pos = -1);

    COMMAND_LINE_ARGUMENTS & Insert (const std::string & arg, INT pos = -1);

    COMMAND_LINE_ARGUMENTS & InsertAsOneToken (const std::string & arg, INT pos = -1);

    INT FindArgument(const std::string& argStr) const;

    BOOL RemoveArguments(const std::string& argStr, INT numOfArgs);

    INT  Argc() const
    {
        return m_argc;
    }

    const CHAR * const * Argv() const
    {
        return m_argv;
    }

    std::string String() const;

    ~COMMAND_LINE_ARGUMENTS()
    {
        Clean();
    }

  private:
    BOOL Enter(INT argc, const CHAR * const * argv, const CHAR * delimiter);

    BOOL Enter(const std::string & commandLine);

    BOOL Clean();

    INT FindArraySize(INT argc, const CHAR * const * argv, const CHAR * delimiter);

    VOID RemoveArgument(INT index);

  private:
    INT m_argc;       //!< size of argv array
    CHAR ** m_argv;   //!< CHAR* array
};

extern VOID SetAddress0x(BOOL val);

extern BOOL CharIsSpace(CHAR c);

extern CHAR CharToUpper(CHAR c);

extern std::string ptrstr(const VOID *val );

extern std::string StringFromAddrint( ADDRINT l);

extern std::string StringFromUint64( UINT64 l);

extern std::string StringDec( UINT64 l, UINT32 digits, CHAR padding);

extern std::string StringDecSigned( INT64 l, UINT32 digits, CHAR padding);

extern std::string StringBignum( INT64 l, UINT32 digits, CHAR padding);

extern std::string Reformat(const std::string& s, const std::string& prefix, UINT32 min_line, UINT32 max_line);

extern std::string StringHex32( UINT32 l, UINT32 digits, BOOL prefix_0x);

extern std::string StringBool(BOOL b);

extern std::string StringTri(TRI t);

extern INT32 Int32FromString(const std::string& s);

extern UINT32 Uint32FromString(const std::string& s);

extern INT64 Int64FromString(const std::string& s);

extern UINT64 Uint64FromString(const std::string& s);

extern FLT64 FLT64FromString(const std::string& s);

extern INT CharToHexDigit(CHAR c);

extern ADDRINT AddrintFromString(const std::string& str);

extern std::string ReadLine(std::istream& inputFile, UINT32 *lineNum);

extern UINT32 Tokenize(const std::string& line, std::string *array, UINT32 n);

inline std::string StringHex( UINT32 l, UINT32 digits, BOOL prefix_0x = TRUE) {return StringHex32(l, digits, prefix_0x);}

inline std::string decstr(INT64 val, UINT32 width=0 )  {return StringDecSigned(val,width,' ');}

inline std::string decstr(INT32 val, UINT32 width=0 )  {return StringDecSigned(val,width,' ');}

inline std::string decstr(INT16 val, UINT32 width=0 )  {return StringDecSigned(val,width,' ');}

inline std::string decstr(UINT64 val, UINT32 width=0 )  {return StringDec(val,width,' ');}

inline std::string decstr(UINT32 val, UINT32 width=0 )  {return StringDec(val,width,' ');}

inline std::string decstr(UINT16 val, UINT32 width=0 )  {return StringDec(val,width,' ');}

inline std::string hexstr(INT64 val, UINT32 width=0 )
{
    std::string ostr;
#if defined(_MSC_VER) && _MSC_VER  >= 1400
    ostr = StringHex( INT32((val >> 16) >> 16), width);
#else
    ostr = StringHex( INT32(val >> 32), width);
#endif
    ostr += StringHex( UINT32(val), 8, FALSE);

    return ostr;
}

inline std::string hexstr(INT32 val, UINT32 width=0 )  {return StringHex(INT32(val),width);}

inline std::string hexstr(INT16 val, UINT32 width=0 )  {return StringHex(INT32(val),width);}

inline std::string hexstr(UINT64 val, UINT32 width=0 )
{
    std::string ostr;
#if defined(_MSC_VER) && _MSC_VER  >= 1400
    ostr = StringHex( UINT32((val >> 16) >> 16), width);
#else
    ostr = StringHex( UINT32(val >> 32), width);
#endif
    ostr += StringHex( UINT32(val), 8, FALSE);

    return ostr;
}

inline std::string hexstr(VOID *p, UINT32 width=0 )
{
#if defined(HOST_IA32E)
    UINT64 val = reinterpret_cast<UINT64>(p);
    return hexstr(val,width);
#else
    UINT32 val = reinterpret_cast<UINT32>(p);
    return StringHex(val,width);
#endif
}

inline std::string hexstr(const VOID *p, UINT32 width=0 )
{
#if defined(HOST_IA32E)
    UINT64 val = reinterpret_cast<UINT64>(p);
    return hexstr(val,width);
#else
    UINT32 val = reinterpret_cast<UINT32>(p);
    return StringHex(val,width);
#endif
}

inline std::string hexstr(UINT32 val, UINT32 width=0 )  {return StringHex(UINT32(val),width);}

inline std::string hexstr(UINT16 val, UINT32 width=0 )  {return StringHex(UINT32(val),width);}

inline std::string ljstr(const std::string& s, UINT32 width, CHAR padding = ' ')
{
    std::string  ostr(width,padding);
    ostr.replace(0,s.length(),s);
    return ostr;
}

struct DECSTR
{
    DECSTR(UINT32 width = 0) :_w(width){}
    template <typename T> std::string operator() (const T & t) const {return decstr(t,_w);}
    UINT32 _w;
};

struct HEXSTR
{
    HEXSTR(UINT32 width = 0) :_w(width){}
    template <typename T> std::string operator() (const T & t) const {return hexstr(t,_w);}
    UINT32 _w;
};

/// @cond INTERNAL_DOXYGEN
template <typename ITER, typename CONV>
    std::string StringSequence(ITER begin, ITER end,
                          const CONV & conv,
                          const std::string & delimiter = std::string(" "))
{
    std::string str;

    for (ITER it = begin;  it != end;  ++it)
    {
        if (it != begin) {str += delimiter;}
        str += conv(*it);
    }
    return str;
}

extern UINT32 BitCount(ADDRINT val);

inline VOID * Addrint2VoidStar(ADDRINT addr)
{
#if defined(HOST_IA32E) && defined(TARGET_IA32)
    ASSERT(false,"Should not be called in cross environment\n");
    return 0;
#else
    return reinterpret_cast<VOID*>(addr);
#endif
}

inline ADDRINT VoidStar2Addrint(const VOID * addr)
{
#if defined(HOST_IA32E) && defined(TARGET_IA32)
    ASSERT(false,"Should not be called in cross environment\n");
    return 0;
#else
    return reinterpret_cast<ADDRINT>(addr);
#endif
}

inline ADDRINT VoidStar2Addrint(VOID * addr)
{
#if defined(HOST_IA32E) && defined(TARGET_IA32)
    ASSERT(false,"Should not be called in cross environment\n");
    return 0;
#else
    return reinterpret_cast<ADDRINT>(addr);
#endif
}

template <typename T> BOOL IsPowerOf2(T value)
{
    return ((value & (value - 1)) == 0);
}

template <typename T> T RoundToNextPower2(T value)
{
    //This algorithm rounds up to the next power of 2 by setting all the significant digits
    //of the number in binary representation to 1. The rest of the digits are set to be 0.
    //It increments the number, thus getting the next power of two. This handles all numbers
    //that aren't powers of two. To handle the case of powers of two, 1 is decremented from
    //the number before the process begins. This makes no difference if the number is not a
    //power of 2, and when it is it makes sure we get the same power of two we had before.
    //For details see "http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2"
    T width = sizeof(T)*8; //bit width
    --value;

    for (T i=1; i < width; i <<= 1) //runs log2(width) times
    {
        value = (value | value >> i); //Turn i^2 bytes to 1.
    }
    ++value;
    return value;
}

template <typename T> T RoundUp(T value, size_t alignment)
{
    if (alignment == 0)
    {
        return value;
    }

    value += alignment - 1;
    value /= alignment;
    value *= alignment;

    return value;
}

template <typename T> T * RoundUp(T * ptr, size_t alignment)
{
    return (reinterpret_cast<T *>(RoundUp(reinterpret_cast<ADDRINT>(ptr), alignment)));
}

template <typename T> T RoundUpPowerOf2(T value, size_t alignment)
{
    if (alignment == 0) return value;
    T align_1 = static_cast<T>(alignment - 1);
    ASSERTX((alignment & align_1) == 0);
    return ((value + align_1) & ~align_1);
}

template <typename T> T * RoundUpPowerOf2(T * ptr, size_t alignment)
{
    return (reinterpret_cast<T *>(RoundUpPowerOf2(reinterpret_cast<ADDRINT>(ptr), alignment)));
}

template <typename T> T RoundDown(T value, size_t alignment)
{
    if (alignment == 0)
    {
        return value;
    }

    value /= alignment;
    value *= alignment;

    return value;
}

template <typename T> T * RoundDown(T * ptr, size_t alignment)
{
    return (reinterpret_cast<T *>(RoundDown(reinterpret_cast<ADDRINT>(ptr), alignment)));
}

template <typename T> T RoundDownPowerOf2(T value, size_t alignment)
{
    if (alignment == 0) return value;
    T align_1 = static_cast<T>(alignment - 1);
    ASSERTX((alignment & align_1) == 0);
    return (value & ~align_1);
}

template <typename T> T * RoundDownPowerOf2(T * ptr, size_t alignment)
{
    return (reinterpret_cast<T *>(RoundDownPowerOf2(reinterpret_cast<ADDRINT>(ptr), alignment)));
}

template<typename T> ADDRINT PointerToInt(const T *ptr)
{
#if (defined(HOST_IA32) && !defined(TARGET_IA32)) || (defined(HOST_IA32E) && !defined(TARGET_IA32E))
    ASSERT(false,"Should not be called in cross environment\n");
    return 0;

#else
    return reinterpret_cast<ADDRINT>(ptr);
#endif
}

template<typename T> T *IntToPointer(ADDRINT addr)
{
#if (defined(HOST_IA32) && !defined(TARGET_IA32)) || (defined(HOST_IA32E) && !defined(TARGET_IA32E))
    ASSERT(false,"Should not be called in cross environment\n");
    return 0;

#else
    return reinterpret_cast<T*>(addr);
#endif
}

inline INT64 VoidStar2Int(VOID * addr)
{
#if defined(HOST_IA32E) && defined(TARGET_IA32)
    ASSERT(false,"Should not be called in cross environment\n");
    return 0;
#else
    return reinterpret_cast<INT64>(addr);
#endif
}

inline BOOL UnsignedImmediateFits(const UINT64 value, const UINT32 bits)
{
    return 0 == (value>>bits);
}

inline void RangeCheck(const UINT32 value, const UINT32 bits)
{
    ASSERT (UnsignedImmediateFits(value, bits),
        decstr(bits) + "-bit range overflow in " + hexstr(value) + "\n");
}

extern BOOL SignedImmediateFits(const INT64 value, const UINT32 origSize, const UINT32 extendedSize);

inline BOOL SignedImmediateFits(const INT64 value, const UINT32 origSize)
{
    return SignedImmediateFits(value, origSize, 8 * sizeof(ADDRINT));
}

extern BOOL HasBaseName(const char *fileName, const char *baseNameToCheckFor);

inline INT32 MEMORY_ReadInt32(ADDRINT  address)
{
    return *reinterpret_cast<INT32*>(address);
}

inline VOID MEMORY_WriteInt32(ADDRINT  address, INT32 value)
{
    *reinterpret_cast<INT32*>(address) = value;
}

inline UINT32 MEMORY_ReadUint32(ADDRINT  address)
{
    return *reinterpret_cast<UINT32*>(address);
}

inline VOID MEMORY_WriteUint32(ADDRINT  address, UINT32 value)
{
    *reinterpret_cast<UINT32*>(address) = value;
}

inline INT64 MEMORY_ReadInt64(ADDRINT  address)
{
    return *reinterpret_cast<INT64*>(address);
}

inline VOID MEMORY_WriteInt64(ADDRINT  address, INT64 value)
{
    *reinterpret_cast<INT64*>(address) = value;
}

inline UINT64 MEMORY_ReadUint64(ADDRINT  address)
{
    return *reinterpret_cast<UINT64*>(address);
}

inline VOID MEMORY_WriteUint64(ADDRINT  address, UINT32 value)
{
    *reinterpret_cast<UINT64*>(address) = value;
}

extern ADDRINT GetPageOfAddr(ADDRINT addr);

#if !defined(TARGET_WINDOWS)
extern ADDRINT ProgramEntry(const ImageLoaderInfo * mainImage, const ImageLoaderInfo * loader);

extern std::string Joinpath(std::string s1, std::string s2);

extern std::string Basename(const std::string& path);

extern CHAR* CreateTmpFileName(const CHAR* fnameTemplate, const UINT32 fnameTemplateSize);

#endif
extern const VOID * GetSp();

inline size_t PtrDiff(const VOID * ptr1, const VOID * ptr2)
{
    return (reinterpret_cast<const INT8 *>(ptr1) - reinterpret_cast<const INT8 *>(ptr2));
}

inline VOID * PtrAtOffset(VOID * ptr, size_t offset)
{
    return (reinterpret_cast<INT8 *>(ptr) + offset);
}

inline const VOID * PtrAtOffset(const VOID * ptr, size_t offset)
{
    return (reinterpret_cast<const INT8 *>(ptr) + offset);
}

template <typename T> T * PtrAtOffset(VOID * ptr, size_t offset)
{
    return (reinterpret_cast<T *>(PtrAtOffset(ptr, offset)));
}

template <typename T> const T * PtrAtOffset(const VOID * ptr, size_t offset)
{
    return (reinterpret_cast<const T *>(PtrAtOffset(ptr, offset)));
}

template <typename T> VOID *
 PushDataToStack(VOID * stack, const T * data, size_t alignment = 1, size_t size = sizeof(T))
{
    stack = PtrAtOffset(stack, 0 - size);
    stack = RoundDown(stack , alignment);
    return memcpy(stack, data, size);
}

inline VOID * PushAddrintToStack(VOID * stack, ADDRINT value)
{
    stack = PtrAtOffset(stack, 0 - sizeof(ADDRINT));
    *reinterpret_cast<ADDRINT *>(stack) = value;
    return stack;
}

class MemRange
{
    public:

    //======= Constructors

    //Default constructor
    MemRange(): m_base(NULL), m_size(0) {}
    //Construct range with a given base address and size
    MemRange(VOID * base, size_t size) : m_base(base), m_size(size) {}
    MemRange(ADDRINT base, size_t size) : m_base(Addrint2VoidStar(base)), m_size(size) {}
    //Construct range with a given base and end addresses.
    //End address of a range is the address of the first byte following the range
    MemRange(VOID * base, VOID * end) : m_base(base), m_size(PtrDiff(end, base)) {}

    //Default copy constructor and assignment operator

    //======= operators == and !=
    BOOL operator == (const MemRange &  range) const
    {
        return ((m_base == range.m_base) && (m_size == range.m_size));
    }
    BOOL operator != (const MemRange &  range) const {return (!(*this == range));}

    //======= Accessors

    //Get/set base address of the range.
    VOID * Base() const {return m_base;}
    MemRange& Base(VOID * base) { m_base = base; return *this;}

    //Get/set size, in bytes, of the range.
    size_t Size() const {return  m_size;}
    MemRange& Size(size_t size) { m_size = size; return *this;}

    //Get end address of the range.
    VOID * End() const {return  PtrAtOffset(m_base, m_size);}

    //Get the last address of the non-empty range.
    VOID * Last() const {return PtrAtOffset(m_base, m_size - 1);}

    /*
    * Check to see whether this range is empty
    *
    * @return  true, if this range is empty, otherwise - false
    */
    BOOL IsEmpty() const {return (m_size == 0);}

    /*
    * Check to see whether this range contains the specified memory address
    *
    * @param[in] addr       memory address, potentially contained in this range
    *
    * @return   true, if this range contains the specified address,
    *           otherwise - false
    */
    BOOL Contains(const VOID * addr) const
    {
        return (PtrDiff(addr, m_base) < m_size);
    }

    /*
    * Check to see whether this range contains the specified memory range
    *
    * @param[in] range      memory range, potentially contained in this range
    *
    * @return   true, if this range contains the specified range,
    *           otherwise - false.
    */
    BOOL Contains(const MemRange & range) const
    {
        return  (Contains(range.m_base) && !range.Contains(End()));
    }

    /* Check to see whether this range intersects with the specified memory range
    *
    * @param[in] range      memory range, potentially intersecting with this range
    *
    * @return   true, if this range intersects with the specified range
    *           otherwise - false
    */
    BOOL Intersects(const MemRange & range) const
    {
        return  (Contains(range.m_base) || range.Contains(m_base));
    }

    /* Check to see whether this range is adjacent to the specified memory range
    *
    * @param[in] range      memory range, potentially adjacent to this range
    *
    * @return   true, if this range is adjacent to the specified range
    *           otherwise - false
    */
    BOOL AdjacentTo(const MemRange & range) const
    {
        return  (((range.m_base == End()) && (range.m_base != 0)) ||
                 ((m_base == range.End()) && (m_base != 0)));
    }

    /* Check to see whether the current stack pointer belongs to this range
    *
    * @return   true, if current stack pointer belongs to this range
    *           otherwise - false
    */
    BOOL IsCurrentStack() const
    {
        return  (Contains(GetSp()));
    }

    /*
    * Round base/end address of the range down/up according to specified alignment
    *
    * @param[in] alignment      alignment value. Must be power of two.
    *
    * @return  reference to aligned range.
    */
    MemRange& Align(UINT32 alignment)
    {
        VOID * end  = RoundUp(End(), alignment);
        m_base = RoundDown(m_base, alignment);
        m_size = PtrDiff(end, m_base);
        return *this;
    }

    /*
    * Round both base and end address of the range up according to specified alignment.
    * The size of the updated range is at least size the range before update.
    * @param[in] alignment      alignment value. Must be power of two.
    *  return  reference to aligned range
    */
    MemRange& AlignUp(UINT32 alignment)
    {
        m_base = RoundUp(m_base, alignment);
        m_size = RoundUp(m_size, alignment);
        return *this;
    }

    /*
    * Round both base and end address of the range down according to specified alignment.
    * The size of the updated range is at least size the range before update.
    * @param[in] alignment      alignment value. Must be power of two.
    * @return  reference to aligned range
    */
    MemRange& AlignDown(UINT32 alignment)
    {
        VOID * end  = RoundDown(End(), alignment);
        m_size = RoundUp(m_size, alignment);
        m_base = PtrAtOffset(end, 0 - m_size);
        return *this;
    }

    /*
    * Shift both base and end address of the range
    * @param[in] offset         shift value.
    * @return  reference to shifted range
    */
    MemRange& Shift(size_t offset)
    {
        m_base = PtrAtOffset(m_base, offset);
        return *this;
    }

    /*
    * Shift base address of the range without changing the end address
    * If the base is shifted to be after the end address, then the region size
    * will be set to zero
    * @param[in] offset         shift value.
    * @return  reference to shifted range
    */
    MemRange& ShiftBase(size_t offset)
    {
        VOID * end  = End();
        m_base = PtrAtOffset(m_base, offset);
        if (end < m_base)
        {
            m_size = 0;
        }
        else
        {
            m_size = PtrDiff(end, m_base);
        }
        return *this;
    }

    /*
    * Shift end address of the range without changing the base address
    * @param[in] offset         shift value.
    * @return  reference to shifted range
    */
    MemRange& ShiftEnd(size_t offset)
    {
        m_size += offset;
        return *this;
    }

    /*
    * Extend boundaries of this range to include the specified range.
    * The new range is the minimal range that contains both ranges.
    * @param[in] range      memory range, to be combined with this range
    * @return  reference to updated range
    */
    MemRange& Combine(const MemRange & range)
    {
        size_t size;
        if (range.m_base >= m_base)
        {
            size = PtrDiff(range.End(), m_base);
        }
        else
        {
            size = PtrDiff(End(), range.m_base);
            *this = range;
        }
        if (size > m_size)
        {
            m_size = size;
        }
        return *this;
    }

    /*
    * Decompose this range into three ranges (each of them may be empty):
    * @param[out]  pIntersection - intersection of <this> range with specified <range>
    * @param[out]  pLowDiff      - part of <this> range, that contains all addresses
    *                              lower than any address in the specified <range>
    * @param[out]  pHighDiff     - part of <this> range, that contains all addresses
    *                              higher than any address in the specified <range>
    * @param[in]   range         - range to be intersected with and subtracted from
    *                              <this> range
    * All three output paramters are optional - the range is not returned if the
    * corresponding parameter is NULL.
    */
    VOID Decompose(const MemRange & range,
                   MemRange * pIntersection,
                   MemRange * pLowDiff,
                   MemRange * pHighDiff) const
    {
        VOID * thisEnd  = End();
        VOID * otherEnd = range.End();

        VOID * intersectionBase;
        VOID * intersectionEnd;

        if (Contains(range.m_base))
        {
            // intersecting ranges
            intersectionBase = range.m_base;
            intersectionEnd  = (range.Contains(thisEnd) ? thisEnd : otherEnd);
        }
        else if (range.Contains(m_base))
        {
            // intersecting ranges
            intersectionBase = m_base;
            intersectionEnd  = (Contains(otherEnd) ? otherEnd : thisEnd);
        }
        else if (m_base > range.m_base)
        {
            // intersection is empty; <this> is higher than <range>
            intersectionBase = intersectionEnd = m_base;
        }
        else
        {
            // intersection is empty; <this> is lower than <range>
            intersectionBase = intersectionEnd = thisEnd;
        }

        if (pIntersection != 0)
        {
            *pIntersection = MemRange(intersectionBase, intersectionEnd);
        }
        if (pLowDiff != 0)
        {
            *pLowDiff = MemRange(m_base, intersectionBase);
        }
        if (pHighDiff != 0)
        {
            *pHighDiff = MemRange(intersectionEnd, thisEnd);
        }
    }

    private:
    VOID * m_base;
    size_t m_size;
};

inline std::string MemRangeToString(const MemRange & range)
{
    return ("[" + hexstr(range.Base()) + " , " + hexstr(range.End()) + "]");
}

extern MemRange MemPageRange(ADDRINT addr);

extern MemRange MemPageRange(const VOID * addr);

class ONCE_STATE
{
  private:
    // Function execution state
    enum
    {
        NO_ONCE,            //!< initial state - function never executed
        ONCE_RUNNING,       //!< function is currently running
        ONCE_SUCCESS,       //!< function executed successfully
        ONCE_FAILURE        //!< function executed but failed
    } m_state;

  public:
    // Initial state - never executed
    ONCE_STATE() : m_state(NO_ONCE) {}

    // Return FALSE if function is already executed. Otherwise, set state to ONCE_RUNNING
    // and return TRUE.
    BOOL Enter()
    {
        if (m_state == NO_ONCE)
        {
            m_state = ONCE_RUNNING;
            return TRUE;
        }
        return FALSE;
    }

    // Set and return exit state of the function just executed
    BOOL Exit(BOOL result = TRUE)
    {
        m_state = (result ? ONCE_SUCCESS : ONCE_FAILURE);
        return result;
    }

    // Return TRUE if function is already executed successfully
    BOOL IsSuccess() const {return (m_state == ONCE_SUCCESS); }
    // Return TRUE if function is already executed but failed
    BOOL IsFailure() const {return (m_state == ONCE_FAILURE); }
    // Return TRUE if function is already executed
    BOOL IsDone() const {return (IsSuccess() || IsFailure()); }

};

struct OBJECT_DELETER
{
    template<typename T> static VOID Delete(T* p) {delete p;}
};

struct ARRAY_DELETER
{
    template<typename T> static VOID Delete(T* p) {delete [] p;}
};

template<typename T, typename D = OBJECT_DELETER, typename C = INT32> class COUNTED_PTR
{
  public:
    COUNTED_PTR() : m_cptr(0){}
    explicit COUNTED_PTR(T * p) : m_cptr((p != 0) ? (new CPTR(p)) : 0) {}
    COUNTED_PTR(const COUNTED_PTR & r) : m_cptr(r.m_cptr) {AddRef();}

    ~COUNTED_PTR() {ReleaseRef();}

    COUNTED_PTR & operator = (const COUNTED_PTR & r)
    {
        r.AddRef();
        ReleaseRef();
        m_cptr = r.m_cptr;
        return *this;
    }

    VOID Reset()
    {
        ReleaseRef();
        m_cptr = 0;
    }

    VOID Reset(T * p)
    {
        if ((m_cptr != 0) && (p == m_cptr->m_ptr)) {return;}
        ReleaseRef();
        m_cptr = ((p != 0) ? (new CPTR(p)) : 0);
    }


    T * Get()           const   {return ((m_cptr != 0) ? m_cptr->m_ptr : 0);}
    T * operator->()    const   {return Get();}
    T & operator*()     const   {return *(Get());}

    BOOL operator == (const COUNTED_PTR &  r) const {return (m_cptr == r.m_cptr);}
    BOOL operator != (const COUNTED_PTR &  r) const {return (m_cptr != r.m_cptr);}
    BOOL operator == (const T *  p) const {return (Get() == p);}
    BOOL operator != (const T *  p) const {return (Get() != p);}

  private:

    struct CPTR
    {
        CPTR(T * p) : m_ptr(p), m_count(1) {}
        T *     m_ptr;
        C       m_count;
    }* m_cptr;

    VOID AddRef() const
    {
        if (m_cptr != 0) {++m_cptr->m_count;}
    }

    VOID ReleaseRef()
    {
        if ((m_cptr != 0) && (--m_cptr->m_count == 0))
        {
            D::Delete(m_cptr->m_ptr);
            delete m_cptr;
            m_cptr = 0;
        }
    }
};

template <typename T> struct OPTIONAL_VALUE
{
    BOOL m_hasValue;    ///< TRUE, if a value has been assigned to this object
    T m_value;          ///< The value assigned to this object

    VOID Set(const T & value)
    {
        m_hasValue = TRUE;
        m_value = value;
    }

    VOID Set(const T * pValue)
    {
        if (pValue != 0)
        {
            Set(*pValue);
        }
        else
        {
            Reset();
        }
    }

    VOID Reset() {m_hasValue = FALSE;}

    BOOL Get(T * pValue) const
    {
        if (m_hasValue && (pValue != 0)) {*pValue = m_value;}
        return m_hasValue;
    }

    BOOL HasValue() const {return m_hasValue;}

    T & Value() {return m_value;}
    const T & Value() const {return m_value;}

    T * ValuePtr() {return ((m_hasValue) ? &m_value : 0);}
    const T * ValuePtr() const {return ((m_hasValue) ? &m_value : 0);}
};

extern STAT_NORM StatRawMmapBytes;

extern STAT_NORM StatReservedBytes;

extern VOID UpdateRawMmapBytes();

/// @endcond
extern std::string StringFlt(FLT64 val , UINT32 precision, UINT32 width);

inline std::string fltstr(FLT64 val, UINT32 prec=0,UINT32 width=0 )  {return StringFlt(val,prec,width);}

struct FLTSTR
{
    FLTSTR(UINT32 prec=0,UINT32 width=0) :_p(prec), _w(width) {}
    template <typename T> std::string operator() (const T & t) const {return fltstr(t,_p,_w);}
    UINT32 _p;
    UINT32 _w;
};

#endif // PIN_G_UTIL_PH


