#ifndef MNN_STB_IMAGE_H
#define MNN_STB_IMAGE_H

#define STBI_FAILURE_USERMSG
#define STB_IMAGE_EXPORTS

#ifdef STB_IMAGE_STATIC
    #define STBIDEF static
    #define STBIWDEF static
#else
    #ifdef _WIN32
        #ifdef STB_IMAGE_EXPORTS
            #ifdef __cplusplus
                #define STBIDEF extern "C" __declspec(dllexport)
                #define STBIWDEF extern "C" __declspec(dllexport)
                #define STBIRDEF extern "C" __declspec(dllexport)
            #else
                #define STBIDEF extern __declspec(dllexport)
                #define STBIWDEF extern __declspec(dllexport)
                #define STBIRDEF extern __declspec(dllexport)
            #endif
        #else
            #ifdef __cplusplus
                #define STBIDEF extern "C" __declspec(dllimport)
                #define STBIWDEF extern "C" __declspec(dllimport)
                #define STBIRDEF extern "C" __declspec(dllimport)
            #else
                #define STBIDEF extern __declspec(dllimport)
                #define STBIWDEF extern __declspec(dllimport)
                #define STBIRDEF extern __declspec(dllimport)
            #endif
        #endif
    #else
        #ifdef __cplusplus
            #define STBIDEF extern "C"
            #define STBIWDEF extern "C"
            #define STBIRDEF extern "C"
        #else
            #define STBIDEF extern
            #define STBIWDEF extern
            #define STBIRDEF extern
        #endif
    #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#endif // MNN_STB_IMAGE_H
