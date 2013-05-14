/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2013, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * \file
 * Common type manipulation (metaprogramming) utilities
 */


/******************************************************************************
 * Common type manipulation (metaprogramming) utilities
 ******************************************************************************/

#pragma once

#include <iostream>

#include "ns_wrapper.cuh"

CUB_NS_PREFIX

/// CUB namespace
namespace cub {



/******************************************************************************
 * Marker types
 ******************************************************************************/

/**
 * \brief A simple "NULL" marker type
 */
struct NullType {};

/** \cond SPECIALIZE */
std::ostream& operator<< (std::ostream& stream, const NullType& val) { return stream; }
/** \endcond */     // SPECIALIZE



/******************************************************************************
 * Static math
 ******************************************************************************/

/**
 * \brief Statically determine log2(N), rounded up.
 *
 * For example:
 *     Log2<8>::VALUE   // 3
 *     Log2<3>::VALUE   // 2
 */
template <int N, int CURRENT_VAL = N, int COUNT = 0>
struct Log2
{
    /// Static logarithm value
    static const int VALUE = Log2<N, (CURRENT_VAL >> 1), COUNT + 1>::VALUE;         // Inductive case
};

/** \cond SPECIALIZE **/
template <int N, int COUNT>
struct Log2<N, 0, COUNT>
{
    static const int VALUE = (1 << (COUNT - 1) < N) ?                               // Base case
        COUNT :
        COUNT - 1;
};
/** \endcond */     // SPECIALIZE




/******************************************************************************
 * Type equality
 ******************************************************************************/

/**
 * \brief Type selection (<tt>IF ? ThenType : ElseType</tt>)
 */
template <bool IF, typename ThenType, typename ElseType>
struct If
{
    /// Conditional type result
    typedef ThenType Type;      // true
};

/** \cond SPECIALIZE **/
template <typename ThenType, typename ElseType>
struct If<false, ThenType, ElseType>
{
    typedef ElseType Type;      // false
};
/** \endcond */     // SPECIALIZE


/******************************************************************************
 * Conditional types
 ******************************************************************************/


/**
 * \brief Type equality test
 */
template <typename A, typename B>
struct Equals
{
    enum {
        VALUE = 0,
        NEGATE = 1
    };
};

/** \cond SPECIALIZE **/
template <typename A>
struct Equals <A, A>
{
    enum {
        VALUE = 1,
        NEGATE = 0
    };
};
/** \endcond */     // SPECIALIZE



/******************************************************************************
 * Qualifier detection
 ******************************************************************************/


/**
 * \brief Volatile modifier test
 */
template <typename Tp>
struct IsVolatile
{
    enum { VALUE = 0 };
};

/** \cond SPECIALIZE **/
template <typename Tp>
struct IsVolatile<Tp volatile>
{
    enum { VALUE = 1 };
};
/** \endcond */     // SPECIALIZE



/******************************************************************************
 * Qualifier removal
 ******************************************************************************/

/**
 * \brief Removes \p const and \p volatile qualifiers from type \p Tp.
 *
 * For example:
 *     <tt>typename RemoveQualifiers<volatile int>::Type         // int;</tt>
 */
template <typename Tp, typename Up = Tp>
struct RemoveQualifiers
{
    /// Type without \p const and \p volatile qualifiers
    typedef Up Type;
};

/** \cond SPECIALIZE **/
template <typename Tp, typename Up>
struct RemoveQualifiers<Tp, volatile Up>
{
    typedef Up Type;
};

template <typename Tp, typename Up>
struct RemoveQualifiers<Tp, const Up>
{
    typedef Up Type;
};

template <typename Tp, typename Up>
struct RemoveQualifiers<Tp, const volatile Up>
{
    typedef Up Type;
};
/** \endcond */     // SPECIALIZE



/******************************************************************************
 * Typedef-detection
 ******************************************************************************/


/**
 * Allows the definition of structures that will detect the presence
 * of the specified type name within other classes
 */
#define CUB_HAS_NESTED_TYPE(detect_struct, nested_type_name)            \
    template <typename T>                                               \
    struct detect_struct                                                \
    {                                                                   \
        template <typename C>                                           \
        static char& test(typename C::nested_type_name*);               \
        template <typename>                                             \
        static int& test(...);                                          \
        enum                                                            \
        {                                                               \
            VALUE = sizeof(test<T>(0)) < sizeof(int)                    \
        };                                                              \
    };



/******************************************************************************
 * Simple enable-if (similar to Boost)
 ******************************************************************************/

/**
 * \brief Simple enable-if (similar to Boost)
 */
template <bool Condition, class T = void>
struct EnableIf
{
    /// Enable-if type for SFINAE dummy variables
    typedef T Type;
};

/** \cond SPECIALIZE **/
template <class T>
struct EnableIf<false, T> {};
/** \endcond */     // SPECIALIZE




/******************************************************************************
 * Simple type traits utilities.
 *
 * For example:
 *     Traits<int>::CATEGORY             // SIGNED_INTEGER
 *     Traits<NullType>::NULL_TYPE       // true
 *     Traits<uint4>::CATEGORY           // NOT_A_NUMBER
 *     Traits<uint4>::PRIMITIVE;         // false
 *
 ******************************************************************************/

/**
 * \brief Basic type traits categories
 */
enum Category
{
    NOT_A_NUMBER,
    SIGNED_INTEGER,
    UNSIGNED_INTEGER,
    FLOATING_POINT
};


/**
 * \brief Basic type traits
 */
template <Category _CATEGORY, bool _PRIMITIVE, bool _NULL_TYPE, typename _UnsignedBits>
struct BaseTraits
{
    /// Category
    static const Category CATEGORY      = _CATEGORY;
    enum
    {
        PRIMITIVE                       = _PRIMITIVE,
        NULL_TYPE                       = _NULL_TYPE
    };
};

/** \cond SPECIALIZE */

/**
 * Basic type traits (unsigned primitive specialization)
 */
template <typename _UnsignedBits>
struct BaseTraits<UNSIGNED_INTEGER, true, false, _UnsignedBits>
{
    typedef _UnsignedBits       UnsignedBits;

    static const Category       CATEGORY    = UNSIGNED_INTEGER;
    static const UnsignedBits   MIN_KEY     = UnsignedBits(0);
    static const UnsignedBits   MAX_KEY     = UnsignedBits(-1);

    enum
    {
        PRIMITIVE = true,
        NULL_TYPE = false
    };


    static __device__ __forceinline__ UnsignedBits TwiddleIn(UnsignedBits key)
    {
        return key;
    }

    static __device__ __forceinline__ UnsignedBits TwiddleOut(UnsignedBits key)
    {
        return key;
    }
};


/**
 * Basic type traits (signed primitive specialization)
 */
template <typename _UnsignedBits>
struct BaseTraits<SIGNED_INTEGER, true, false, _UnsignedBits>
{
    typedef _UnsignedBits       UnsignedBits;

    static const Category       CATEGORY    = SIGNED_INTEGER;
    static const UnsignedBits   HIGH_BIT    = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
    static const UnsignedBits   MIN_KEY     = HIGH_BIT;
    static const UnsignedBits   MAX_KEY     = UnsignedBits(-1) ^ HIGH_BIT;

    enum
    {
        PRIMITIVE = true,
        NULL_TYPE = false
    };

    static __device__ __forceinline__ UnsignedBits TwiddleIn(UnsignedBits key)
    {
        return key ^ HIGH_BIT;
    };

    static __device__ __forceinline__ UnsignedBits TwiddleOut(UnsignedBits key)
    {
        return key ^ HIGH_BIT;
    };

};


/**
 * Basic type traits (fp primitive specialization)
 */
template <typename _UnsignedBits>
struct BaseTraits<FLOATING_POINT, true, false, _UnsignedBits>
{
    typedef _UnsignedBits       UnsignedBits;

    static const Category       CATEGORY    = FLOATING_POINT;
    static const UnsignedBits   HIGH_BIT    = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
    static const UnsignedBits   MIN_KEY     = UnsignedBits(-1);
    static const UnsignedBits   MAX_KEY     = UnsignedBits(-1) ^ HIGH_BIT;

    static __device__ __forceinline__ UnsignedBits TwiddleIn(UnsignedBits key)
    {
        UnsignedBits mask = (key & HIGH_BIT) ? UnsignedBits(-1) : HIGH_BIT;
        return key ^ mask;
    };

    static __device__ __forceinline__ UnsignedBits TwiddleOut(UnsignedBits key)
    {
        UnsignedBits mask = (key & HIGH_BIT) ? HIGH_BIT : UnsignedBits(-1);
        return key ^ mask;
    };

    enum
    {
        PRIMITIVE = true,
        NULL_TYPE = false
    };
};
/** \endcond */     // SPECIALIZE


/**
 * \brief Numeric type traits
 */
template <typename T> struct NumericTraits :            BaseTraits<NOT_A_NUMBER, false, false, T> {};

/** \cond SPECIALIZE */
template <> struct NumericTraits<NullType> :            BaseTraits<NOT_A_NUMBER, false, true, NullType> {};

template <> struct NumericTraits<char> :                BaseTraits<SIGNED_INTEGER, true, false, unsigned char> {};
template <> struct NumericTraits<signed char> :         BaseTraits<SIGNED_INTEGER, true, false, unsigned char> {};
template <> struct NumericTraits<short> :               BaseTraits<SIGNED_INTEGER, true, false, unsigned short> {};
template <> struct NumericTraits<int> :                 BaseTraits<SIGNED_INTEGER, true, false, unsigned int> {};
template <> struct NumericTraits<long> :                BaseTraits<SIGNED_INTEGER, true, false, unsigned long> {};
template <> struct NumericTraits<long long> :           BaseTraits<SIGNED_INTEGER, true, false, unsigned long long> {};

template <> struct NumericTraits<unsigned char> :       BaseTraits<UNSIGNED_INTEGER, true, false, unsigned char> {};
template <> struct NumericTraits<unsigned short> :      BaseTraits<UNSIGNED_INTEGER, true, false, unsigned short> {};
template <> struct NumericTraits<unsigned int> :        BaseTraits<UNSIGNED_INTEGER, true, false, unsigned int> {};
template <> struct NumericTraits<unsigned long> :       BaseTraits<UNSIGNED_INTEGER, true, false, unsigned long> {};
template <> struct NumericTraits<unsigned long long> :  BaseTraits<UNSIGNED_INTEGER, true, false, unsigned long long> {};

template <> struct NumericTraits<float> :               BaseTraits<FLOATING_POINT, true, false, unsigned int> {};
template <> struct NumericTraits<double> :              BaseTraits<FLOATING_POINT, true, false, unsigned long long> {};
/** \endcond */     // SPECIALIZE


/**
 * \brief Type traits
 */
template <typename T>
struct Traits : NumericTraits<typename RemoveQualifiers<T>::Type> {};



/******************************************************************************
 * Simple array traits utilities.
 *
 * For example:
 *
 *     typedef int A[10];
 *     ArrayTraits<A>::DIMS             // 1
 *     ArrayTraits<A>::ELEMENTS         // 10
 *     typename ArrayTraits<A>::Type    // int
 *
 *     typedef int B[10][20];
 *     ArrayTraits<B>::DIMS             // 2
 *     ArrayTraits<B>::ELEMENTS         // 200
 *     typename ArrayTraits<B>::Type    // int
 *
 *     typedef int C;
 *     ArrayTraits<C>::DIMS             // 0
 *     ArrayTraits<C>::ELEMENTS         // 1
 *     typename ArrayTraits<C>::Type    // int

 *     typedef int* D;
 *     ArrayTraits<D>::DIMS             // 1
 *     ArrayTraits<D>::ELEMENTS         // 1
 *     typename ArrayTraits<D>::Type    // int
 *
 *     typedef int (*E)[2];
 *     ArrayTraits<E>::DIMS             // 2
 *     ArrayTraits<E>::ELEMENTS         // 2
 *     typename ArrayTraits<E>::Type    // int
 *
 ******************************************************************************/

/**
 * \brief Array traits
 */
template <typename ArrayType, int LENGTH = -1>
struct ArrayTraits;

/** \cond SPECIALIZE */

/**
 * Specialization for non array type
 */
template <typename DimType, int LENGTH>
struct ArrayTraits
{
    typedef DimType Type;

    enum {
        ELEMENTS    = 1,
        DIMS        = 0
    };
};


/**
 * Specialization for pointer type
 */
template <typename DimType, int LENGTH>
struct ArrayTraits<DimType*, LENGTH>
{
    typedef typename ArrayTraits<DimType>::Type Type;

    enum {
        ELEMENTS    = ArrayTraits<DimType>::ELEMENTS,
        DIMS        = ArrayTraits<DimType>::DIMS + 1,
    };
};


/**
 * Specialization for array type
 */
template <typename DimType, int LENGTH>
struct ArrayTraits<DimType[LENGTH], -1>
{
    typedef typename ArrayTraits<DimType>::Type Type;

    enum {
        ELEMENTS    = ArrayTraits<DimType>::ELEMENTS * LENGTH,
        DIMS        = ArrayTraits<DimType>::DIMS + 1,
    };
};

/** \endcond */     // SPECIALIZE

} // namespace cub
CUB_NS_POSTFIX
