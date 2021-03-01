//Groups: @ingroup\s+(API_REF|KNOBS|IMG_BASIC_API|INS_BASIC_API|INS_INST_API|INS_BASIC_API_GEN_IA32|INS_BASIC_API_IA32|INS_MOD_API_GEN_IA32|SEC_BASIC_API|RTN_BASIC_API|REG_BASIC_API|REG_CPU_GENERIC|REG_CPU_IA32|TRACE_BASIC_API|BBL_BASIC_API|SYM_BASIC_API|MISC_PRINT|MISC_PARSE|KNOB_API|KNOB_BASIC|KNOB_PRINT|LOCK|PIN_CONTROL|TRACE_VERSION_API|BUFFER_API|PROTO_API|PIN_PROCESS_API|PIN_THREAD_API|PIN_SYSCALL_API|WINDOWS_SYSCALL_API_UNDOC|DEBUG_API|ERROR_FILE_BASIC|TYPE_BASE|INSTLIB|ALARM|CHILD_PROCESS_API|UTILS|MISC|CONTEXT_API|PHYSICAL_CONTEXT_API|PIN_CALLBACKS|EXCEPTION_API|APPDEBUG_API|STOPPED_THREAD_API|BUFFER_API|PROTO|INST_ARGS|DEPRECATED_PIN_API|INTERNAL_EXCEPTION_PRIVATE_UNDOCUMENTED|PIN_THREAD_PRIVATE|CHILD_PROCESS_INTERNAL|BBL_BASIC|ROGUE_BASIC_API|MESSAGE_TYPE|MESSAGE_BASIC|ERRFILE|MISC_BASIC|ITC_INST_API|CONTEXT_API_UNDOC|EXCEPTION_API_UNDOC|UNDOCUMENTED_PIN_API|OPIN|TRACE_VERSIONS

#ifdef PIN_G_IMAGE_PH
#error duplicate inclusion of image
#else
#define PIN_G_IMAGE_PH
/// @cond INTERNAL_DOXYGEN
extern IMG IMG_CreateAt(const char* filename, ADDRINT start, USIZE size, ADDRINT loadOffset, BOOL mainExecutable);

extern VOID IMG_ReplayImageLoad(IMG img);

extern VOID* IMG_GetLoaderInfo(IMG img);

extern VOID IMG_SetLoaderInfo(IMG img, VOID* loaderInfo);

/// @endcond
extern IMG IMG_Next(IMG img);

extern IMG IMG_Prev(IMG img);

extern IMG IMG_Invalid();

extern BOOL IMG_Valid(IMG img);

extern SEC IMG_SecHead(IMG img);

extern SEC IMG_SecTail(IMG img);

extern SYM IMG_RegsymHead(IMG img);

extern PIN_DEPRECATED_API ADDRINT IMG_Entry(IMG img);

extern ADDRINT IMG_EntryAddress(IMG img);

extern BOOL IMG_HasProperty(IMG img, IMG_PROPERTY property);

extern const std::string& IMG_Name(IMG img);

extern ADDRINT IMG_Gp(IMG img);

extern ADDRINT IMG_LoadOffset(IMG img);

extern ADDRINT IMG_LowAddress(IMG img);

extern ADDRINT IMG_HighAddress(IMG img);

extern  ADDRINT IMG_StartAddress(IMG img);

extern USIZE IMG_SizeMapped(IMG img);

extern IMG_TYPE IMG_Type(IMG img);

extern BOOL IMG_IsMainExecutable(IMG x);

extern BOOL IMG_hasLinesData(IMG x);

extern BOOL IMG_IsInterpreter(IMG x);

extern BOOL IMG_IsStaticExecutable(IMG x);

extern BOOL IMG_IsVDSO(IMG img);

extern UINT32 IMG_NumRegions(IMG img);

extern ADDRINT IMG_RegionHighAddress(IMG img, UINT32 n);

extern ADDRINT IMG_RegionLowAddress(IMG img, UINT32 n);

extern IMG SEC_Img(SEC sec);

extern SEC SEC_Next(SEC sec);

extern SEC SEC_Prev(SEC sec);

extern SEC SEC_Invalid();

extern BOOL SEC_Valid(SEC x);

extern RTN SEC_RtnHead(SEC sec);

extern RTN SEC_RtnTail(SEC sec);

extern const std::string& SEC_Name(SEC sec);

extern SEC_TYPE SEC_Type(SEC sec);

extern BOOL SEC_Mapped(SEC sec);

extern const VOID *SEC_Data(SEC sec);

extern SEC RTN_Sec(RTN x);

extern RTN RTN_Next(RTN x);

extern RTN RTN_Prev(RTN x);

extern RTN RTN_Invalid();

extern BOOL RTN_Valid(RTN x);

extern const std::string& RTN_Name(RTN x);

extern BOOL RTN_IsArtificial(RTN x);

extern SYM RTN_Sym(RTN x);

extern UINT RTN_DynamicMethodId(RTN x);

extern AFUNPTR RTN_Funptr(RTN x);

extern UINT32 RTN_Id(RTN x);

extern INS BBL_InsHead(BBL x);

extern INS BBL_InsTail(BBL x);

extern BBL BBL_Next(BBL x);

extern BBL BBL_Prev(BBL x);

extern BOOL BBL_Valid(BBL x);

extern RTN INS_Rtn(INS x);

extern INS INS_Next(INS x);

extern INS INS_Prev(INS x);

extern INS INS_Invalid();

extern BOOL INS_Valid(INS x);

extern ADDRINT INS_Address(INS ins);

extern USIZE INS_Size(INS ins);

extern PIN_DEPRECATED_API ADDRINT INS_DirectBranchOrCallTargetAddress(INS ins);

extern ADDRINT INS_DirectControlFlowTargetAddress(INS ins);

extern ADDRINT INS_NextAddress(INS ins);

extern SYM SYM_Next(SYM x);

extern SYM SYM_Prev(SYM x);

extern const std::string& SYM_Name(SYM x);

extern SYM SYM_Invalid();

extern BOOL SYM_Valid(SYM x);

extern BOOL SYM_Dynamic(SYM x);

extern BOOL SYM_GeneratedByPin(SYM x);

extern BOOL SYM_IFuncImplementation(SYM x);

extern BOOL SYM_IFuncResolver(SYM x);

extern ADDRINT SYM_Value(SYM x);

extern UINT32 SYM_Index(SYM x);

extern ADDRINT SYM_Address(SYM x);

typedef VOID (*IMAGECALLBACK)(IMG, VOID *);

enum UNDECORATION {
    UNDECORATION_COMPLETE,  ///< Undecorate to full symbol signature
    UNDECORATION_NAME_ONLY  ///< Undecorate to [scope::]name
};

enum SYMBOL_INFO_MODE {
    NO_SYMBOLS     = 0,       ///< No symbols required
    EXPORT_SYMBOLS = (1<<0),  ///< Only symbols taken from export table are provided (Windows only)
    DEBUG_SYMBOLS  = (1<<1),  ///< Debug symbols (Windows only, currently implemented as DEBUG_OR_EXPORT_SYMBOLS)
    IFUNC_SYMBOLS  = (1<<2),  ///< IFUNC symbols (Linux only)
    DEBUG_OR_EXPORT_SYMBOLS = (DEBUG_SYMBOLS | EXPORT_SYMBOLS) ///< First debug symbols if available,
                                                               ///<  otherwise export symbols (Windows only)
};

class SymbolAddressRange
{
    public:
        SymbolAddressRange() : base(0), size(0), name("") {}
        SymbolAddressRange(std::string sym_name, ADDRINT sym_base, size_t sym_size) :
            base(sym_base), size(sym_size), name(sym_name) {}
        SymbolAddressRange(const SymbolAddressRange &obj) {base=obj.base; size=obj.size; name=obj.name;}
        SymbolAddressRange& operator = (const SymbolAddressRange &obj) {base=obj.base; size=obj.size; name=obj.name; return *this;}
        BOOL operator < (const SymbolAddressRange &  range) const {return (base < range.base);}
        BOOL Contains(const ADDRINT addr) const { return ((addr-base)<size); }
    public:
        ADDRINT base;
        size_t size;
        std::string name;
};

struct SymbolDebugInfo
{
    SymbolDebugInfo(std::string sym_name, ADDRINT sym_base, size_t sym_size) :
        name(sym_name), range(sym_name, sym_base, sym_size) {}
    std::string name;
    SymbolAddressRange range;
};

extern UINT32 IMG_Id(IMG x);

extern VOID* IMG_DynamicRawData(IMG img);

extern IMG IMG_FindImgById(UINT32 id);

extern IMG IMG_FindByAddress(ADDRINT address);

extern PIN_CALLBACK RTN_AddInstrumentFunction(RTN_INSTRUMENT_CALLBACK fun, VOID *val);

extern PIN_CALLBACK IMG_AddInstrumentFunction(IMAGECALLBACK fun, VOID *v);

extern PIN_CALLBACK IMG_AddUnloadFunction(IMAGECALLBACK fun, VOID *v);

extern USIZE RTN_Range(RTN rtn);

extern USIZE RTN_Size(RTN rtn);

extern RTN RTN_IFuncResolver(RTN rtn);

extern RTN RTN_IFuncImplementation(RTN rtn);

extern IMG IMG_Open(const std::string& filename);

extern VOID IMG_Close(IMG img);

extern std::string RTN_FindNameByAddress(ADDRINT address);

extern RTN RTN_FindByAddress(ADDRINT address);

extern RTN RTN_FindByName(IMG img, const CHAR * name);

extern VOID RTN_Open(RTN rtn);

extern VOID RTN_Close(RTN rtn);

extern INS RTN_InsHead(RTN rtn);

extern INS RTN_InsHeadOnly(RTN rtn);

extern INS RTN_InsTail(RTN rtn);

extern UINT32 RTN_NumIns(RTN rtn);

extern VOID RTN_InsertCall(RTN rtn, IPOINT action, AFUNPTR funptr, ...);

extern ADDRINT RTN_Address(RTN rtn);

extern RTN RTN_CreateAt(ADDRINT address, std::string name);

extern BOOL RTN_IsDynamic(RTN rtn);

extern ADDRINT SEC_Address(SEC sec);

extern BOOL SEC_IsReadable(SEC sec);

extern BOOL SEC_IsWriteable(SEC sec);

extern BOOL SEC_IsExecutable(SEC sec);

extern USIZE SEC_Size(SEC sec);

extern BOOL BBL_Original(BBL bbl);

extern ADDRINT BBL_Address(BBL bbl);

extern USIZE BBL_Size(BBL bbl);

extern IMG APP_ImgHead();

extern IMG APP_ImgTail();

extern std::string PIN_UndecorateSymbolName(const std::string & symbolName, UNDECORATION style);

#endif // PIN_G_IMAGE_PH


