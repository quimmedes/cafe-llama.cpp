#include "common.h"
#include "dequantize.h"
#include "quantize.h"

template<typename T0, typename T1>
kernel void kernel_cpy_t_t(
        constant ggml_metal_kargs_cpy & args,
        device  const char * src0,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig[2];
    const int32_t i02 = tgpig[1];
    const int32_t i01 = ntg[1] == 1 ? tgpig[0]%args.ne01 : tgpig[0]*ntg[1] + tpitg.y;
    const int32_t iw0 = ntg[1] == 1 ? tgpig[0]/args.ne01 : 0;

    if (i01 >= args.ne01) {
        return;
    }

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int32_t i3 = n/(args.ne2*args.ne1*args.ne0);
    const int32_t i2 = (n - i3*args.ne2*args.ne1*args.ne0)/(args.ne1*args.ne0);
    const int32_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0)/args.ne0;
    const int32_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0);

    device T1 * dst_data = (device T1 *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

    for (int32_t i00 = iw0*ntg[0] + tpitg.x; i00 < args.ne00;) {
        device const T0 * src = (device T0 *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + i00*args.nb00);
        dst_data[i00] = (T1) src[0];
        break;
    }
}

typedef decltype(kernel_cpy_t_t<float, float>) kernel_cpy_t;

template [[host_name("kernel_cpy_f32_f32")]]   kernel kernel_cpy_t kernel_cpy_t_t<float,   float>;
template [[host_name("kernel_cpy_f32_f16")]]   kernel kernel_cpy_t kernel_cpy_t_t<float,   half>;
template [[host_name("kernel_cpy_f32_i32")]]   kernel kernel_cpy_t kernel_cpy_t_t<float,   int32_t>;
template [[host_name("kernel_cpy_i32_f32")]]   kernel kernel_cpy_t kernel_cpy_t_t<int32_t, float>;
template [[host_name("kernel_cpy_i32_i32")]]   kernel kernel_cpy_t kernel_cpy_t_t<int32_t, int32_t>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_cpy_f32_bf16")]]  kernel kernel_cpy_t kernel_cpy_t_t<float,   bfloat>;
#endif
template [[host_name("kernel_cpy_f16_f32")]]   kernel kernel_cpy_t kernel_cpy_t_t<half,    float>;
template [[host_name("kernel_cpy_f16_f16")]]   kernel kernel_cpy_t kernel_cpy_t_t<half,    half>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_cpy_bf16_f32")]]  kernel kernel_cpy_t kernel_cpy_t_t<bfloat,  float>;
template [[host_name("kernel_cpy_bf16_bf16")]] kernel kernel_cpy_t kernel_cpy_t_t<bfloat,  bfloat>;
#endif
template [[host_name("kernel_cpy_turbo4_turbo4")]] kernel kernel_cpy_t kernel_cpy_t_t<block_turbo4_0, block_turbo4_0>;
template [[host_name("kernel_cpy_turbo2_turbo2")]] kernel kernel_cpy_t kernel_cpy_t_t<block_turbo2_0, block_turbo2_0>;
template [[host_name("kernel_cpy_turbo3_turbo3")]] kernel kernel_cpy_t kernel_cpy_t_t<block_turbo3_0, block_turbo3_0>;

template<short QK,
         typename block_q,
         void (*quantize_func)(device const float *, device block_q &)>
kernel void kernel_cpy_f32_q(
        constant ggml_metal_kargs_cpy & args,
        device const char * src0,
        device char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig[2];
    const int32_t i02 = tgpig[1];
    const int32_t i01 = ntg[1] == 1 ? tgpig[0]%args.ne01 : tgpig[0]*ntg[1] + tpitg.y;
    const int32_t iw0 = ntg[1] == 1 ? tgpig[0]/args.ne01 : 0;

    if (i01 >= args.ne01) {
        return;
    }

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int32_t i3 = n / (args.ne2*args.ne1*args.ne0);
    const int32_t i2 = (n - i3*args.ne2*args.ne1*args.ne0) / (args.ne1*args.ne0);
    const int32_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0) / args.ne0;
    const int32_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0)/QK;

    device block_q * dst_data = (device block_q *)(dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

    for (int32_t i00 = iw0*ntg[0] + tpitg.x; i00 < args.nk0;) {
        device const float * src = (device const float *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + (i00*QK)*args.nb00);

        quantize_func(src, dst_data[i00]);

        break;
    }
}

typedef decltype(kernel_cpy_f32_q<QK8_0,  block_q8_0,  quantize_q8_0>)  cpy_f_q_t;

template [[host_name("kernel_cpy_f32_q8_0")]]   kernel cpy_f_q_t kernel_cpy_f32_q<QK8_0,  block_q8_0,   quantize_q8_0>;
template [[host_name("kernel_cpy_f32_q1_0")]]   kernel cpy_f_q_t kernel_cpy_f32_q<QK1_0,  block_q1_0,   quantize_q1_0>;
template [[host_name("kernel_cpy_f32_q2_0")]]   kernel cpy_f_q_t kernel_cpy_f32_q<QK2_0,  block_q2_0,   quantize_q2_0>;
template [[host_name("kernel_cpy_f32_q4_0")]]   kernel cpy_f_q_t kernel_cpy_f32_q<QK4_0,  block_q4_0,   quantize_q4_0>;
template [[host_name("kernel_cpy_f32_q4_1")]]   kernel cpy_f_q_t kernel_cpy_f32_q<QK4_1,  block_q4_1,   quantize_q4_1>;
template [[host_name("kernel_cpy_f32_q5_0")]]   kernel cpy_f_q_t kernel_cpy_f32_q<QK5_0,  block_q5_0,   quantize_q5_0>;
template [[host_name("kernel_cpy_f32_q5_1")]]   kernel cpy_f_q_t kernel_cpy_f32_q<QK5_1,  block_q5_1,   quantize_q5_1>;
template [[host_name("kernel_cpy_f32_iq4_nl")]] kernel cpy_f_q_t kernel_cpy_f32_q<QK4_NL, block_iq4_nl, quantize_iq4_nl>;
template [[host_name("kernel_cpy_f32_tq2_0")]]  kernel cpy_f_q_t kernel_cpy_f32_q<QK_K,   block_tq2_0,  quantize_tq2_0>;

template<typename T4x4, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread T4x4 &)>
kernel void kernel_cpy_q_f32(
        constant ggml_metal_kargs_cpy & args,
        device  const char * src0,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig[2];
    const int32_t i02 = tgpig[1];
    const int32_t i01 = ntg[1] == 1 ? tgpig[0]%args.ne01 : tgpig[0]*ntg[1] + tpitg.y;
    const int32_t iw0 = ntg[1] == 1 ? tgpig[0]/args.ne01 : 0;

    if (i01 >= args.ne01) {
        return;
    }

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int32_t i3 = n/(args.ne2*args.ne1*args.ne0);
    const int32_t i2 = (n - i3*args.ne2*args.ne1*args.ne0)/(args.ne1*args.ne0);
    const int32_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0)/args.ne0;
    const int32_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0);

    device const block_q * src_data = (device const block_q *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device       T4x4    * dst_data = (device       T4x4    *)(dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1 + i0*args.nb0);

    for (int32_t i00 = iw0*ntg[0] + tpitg.x; i00 < args.nk0;) {
        T4x4 temp;
        dequantize_func(src_data + i00/nl, i00%nl, temp);
        dst_data[i00] = temp;

        break;
    }
}

typedef decltype(kernel_cpy_q_f32<float4x4, block_q4_0, 2, dequantize_q4_0>) cpy_q_f_t;

template [[host_name("kernel_cpy_q1_0_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_q1_0, 8, dequantize_q1_0>;
template [[host_name("kernel_cpy_q2_0_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_q2_0, 4, dequantize_q2_0>;
template [[host_name("kernel_cpy_q4_0_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_q4_0, 2, dequantize_q4_0>;
template [[host_name("kernel_cpy_q4_1_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_q4_1, 2, dequantize_q4_1>;
template [[host_name("kernel_cpy_q5_0_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_q5_0, 2, dequantize_q5_0>;
template [[host_name("kernel_cpy_q5_1_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_q5_1, 2, dequantize_q5_1>;
template [[host_name("kernel_cpy_q8_0_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_q8_0, 2, dequantize_q8_0>;

template [[host_name("kernel_cpy_tq2_0_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_tq2_0, QK_NL, dequantize_tq2_0>;

template [[host_name("kernel_cpy_q1_0_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_q1_0, 8, dequantize_q1_0>;
template [[host_name("kernel_cpy_q2_0_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_q2_0, 4, dequantize_q2_0>;
template [[host_name("kernel_cpy_q4_0_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_q4_0, 2, dequantize_q4_0>;
template [[host_name("kernel_cpy_q4_1_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_q4_1, 2, dequantize_q4_1>;
template [[host_name("kernel_cpy_q5_0_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_q5_0, 2, dequantize_q5_0>;
template [[host_name("kernel_cpy_q5_1_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_q5_1, 2, dequantize_q5_1>;
template [[host_name("kernel_cpy_q8_0_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_q8_0, 2, dequantize_q8_0>;

template [[host_name("kernel_cpy_tq2_0_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_tq2_0, QK_NL, dequantize_tq2_0>;
template [[host_name("kernel_cpy_turbo4_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_turbo4_0, 8, dequantize_turbo4_0>;
template [[host_name("kernel_cpy_turbo2_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_turbo2_0, 2, dequantize_turbo2_0>;
template [[host_name("kernel_cpy_turbo3_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_turbo3_0, 2, dequantize_turbo3_0>;

template [[host_name("kernel_cpy_turbo4_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_turbo4_0, 8, dequantize_turbo4_0>;
template [[host_name("kernel_cpy_turbo2_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_turbo2_0, 2, dequantize_turbo2_0>;
template [[host_name("kernel_cpy_turbo3_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_turbo3_0, 2, dequantize_turbo3_0>;

template<typename T>
kernel void kernel_concat(
        constant ggml_metal_kargs_concat & args,
        device  const char * src0,
        device  const char * src1,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {

    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = ntg.y == 1 ? tgpig.x : tgpig.x*ntg.y + tpitg.y;

    if (i1 >= args.ne1) {
        return;
    }

    int o[4] = {0, 0, 0, 0};
    o[args.dim] = args.dim == 0 ? args.ne00 : (args.dim == 1 ? args.ne01 : (args.dim == 2 ? args.ne02 : args.ne03));

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        device const T * x;

        if (i0 < args.ne00 && i1 < args.ne01 && i2 < args.ne02 && i3 < args.ne03) {
            x = (device const T *)(src0 + (i3       )*args.nb03 + (i2       )*args.nb02 + (i1       )*args.nb01 + (i0       )*args.nb00);
        } else {
            x = (device const T *)(src1 + (i3 - o[3])*args.nb13 + (i2 - o[2])*args.nb12 + (i1 - o[1])*args.nb11 + (i0 - o[0])*args.nb10);
        }

        device T * y = (device T *)(dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

        *y = *x;
    }
}

typedef decltype(kernel_concat<float>) kernel_concat_t;

template [[host_name("kernel_concat_f32")]]  kernel kernel_concat_t kernel_concat<float>;
template [[host_name("kernel_concat_f16")]]  kernel kernel_concat_t kernel_concat<half>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_concat_bf16")]] kernel kernel_concat_t kernel_concat<bfloat>;
#endif
template [[host_name("kernel_concat_i8")]]   kernel kernel_concat_t kernel_concat<char>;
template [[host_name("kernel_concat_i16")]]  kernel kernel_concat_t kernel_concat<short>;
template [[host_name("kernel_concat_i32")]]  kernel kernel_concat_t kernel_concat<int>;
template [[host_name("kernel_concat_i64")]]  kernel kernel_concat_t kernel_concat<long>;

template<typename block_q>
kernel void kernel_concat_q(
        constant ggml_metal_kargs_concat & args,
        device  const char * src0,
        device  const char * src1,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {

    // note: for quantized types, the args are in units of blocks (nb0 == type_size)
    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = ntg.y == 1 ? tgpig.x : tgpig.x*ntg.y + tpitg.y;

    if (i1 >= args.ne1) {
        return;
    }

    int o[4] = {0, 0, 0, 0};
    o[args.dim] = args.dim == 0 ? args.ne00 : (args.dim == 1 ? args.ne01 : (args.dim == 2 ? args.ne02 : args.ne03));

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        device const block_q * x;

        if (i0 < args.ne00 && i1 < args.ne01 && i2 < args.ne02 && i3 < args.ne03) {
            x = (device const block_q *)(src0 + (i3       )*args.nb03 + (i2       )*args.nb02 + (i1       )*args.nb01 + (i0       )*args.nb00);
        } else {
            x = (device const block_q *)(src1 + (i3 - o[3])*args.nb13 + (i2 - o[2])*args.nb12 + (i1 - o[1])*args.nb11 + (i0 - o[0])*args.nb10);
        }

        device block_q * y = (device block_q *)(dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

        *y = *x;
    }
}

typedef decltype(kernel_concat_q<block_q4_0>) kernel_concat_q_t;

template [[host_name("kernel_concat_q4_0")]] kernel kernel_concat_q_t kernel_concat_q<block_q4_0>;
template [[host_name("kernel_concat_q4_1")]] kernel kernel_concat_q_t kernel_concat_q<block_q4_1>;
template [[host_name("kernel_concat_q5_0")]] kernel kernel_concat_q_t kernel_concat_q<block_q5_0>;
template [[host_name("kernel_concat_q5_1")]] kernel kernel_concat_q_t kernel_concat_q<block_q5_1>;
template [[host_name("kernel_concat_q8_0")]] kernel kernel_concat_q_t kernel_concat_q<block_q8_0>;

template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread float4x4 &)>
kernel void kernel_get_rows_q(
        constant ggml_metal_kargs_get_rows & args,
        device const void * src0,
        device const void * src1,
        device       void * dst,
        uint3               tgpig[[threadgroup_position_in_grid]],
        ushort              tiitg[[thread_index_in_threadgroup]],
        ushort3             ntg  [[threads_per_threadgroup]]) {
    const int32_t iw0 = tgpig.x/args.ne10;
    const int32_t i10 = tgpig.x%args.ne10;
    const int32_t i11 = tgpig.y;
    const int32_t i12 = tgpig.z;

    const int32_t r = ((const device int32_t *) ((const device char *) src1 + i12*args.nb12 + i11*args.nb11 + i10*args.nb10))[0];

    const int32_t i02 = i11;
    const int32_t i03 = i12;

    auto psrc = (device const block_q *) ((const device char *) src0 + i03*args.nb03 + i02*args.nb02 +   r*args.nb01);
    auto pdst = (device      float4x4 *) ((      device char *) dst  + i12*args.nb3  + i11*args.nb2  + i10*args.nb1);

    for (int ind = iw0*ntg.x + tiitg; ind < args.ne00t;) {
        float4x4 temp;
        dequantize_func(psrc + ind/nl, ind%nl, temp);
        pdst[ind] = temp;

        break;
    }
}

template<typename T0, typename T>
kernel void kernel_get_rows_f(
        constant ggml_metal_kargs_get_rows & args,
        device const void * src0,
        device const void * src1,
        device       void * dst,
        uint3               tgpig[[threadgroup_position_in_grid]],
        ushort              tiitg[[thread_index_in_threadgroup]],
        ushort3             ntg [[threads_per_threadgroup]]) {
    const int32_t iw0 = tgpig.x/args.ne10;
    const int32_t i10 = tgpig.x%args.ne10;
    const int32_t i11 = tgpig.y;
    const int32_t i12 = tgpig.z;

    const int32_t r = ((const device int32_t *) ((const device char *) src1 + i12*args.nb12 + i11*args.nb11 + i10*args.nb10))[0];

    const int32_t i02 = i11;
    const int32_t i03 = i12;

    auto psrc = (const device T0 *) ((const device char *) src0 + i03*args.nb03 + i02*args.nb02 +   r*args.nb01);
    auto pdst = (      device T  *) ((      device char *)  dst + i12*args.nb3  + i11*args.nb2  + i10*args.nb1);

    for (int ind = iw0*ntg.x + tiitg; ind < args.ne00t;) {
        pdst[ind] = psrc[ind];

        break;
    }
}

typedef decltype(kernel_get_rows_f<float, float>) get_rows_f_t;

template [[host_name("kernel_get_rows_f32")]]  kernel get_rows_f_t kernel_get_rows_f<float, float>;
template [[host_name("kernel_get_rows_f16")]]  kernel get_rows_f_t kernel_get_rows_f<half,  float>;
template [[host_name("kernel_get_rows_i32")]]  kernel get_rows_f_t kernel_get_rows_f<int32_t, int32_t>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_get_rows_bf16")]] kernel get_rows_f_t kernel_get_rows_f<bfloat, float>;
#endif

typedef decltype(kernel_get_rows_q<block_q4_0, 2, dequantize_q4_0>) get_rows_q_t;

template [[host_name("kernel_get_rows_q1_0")]]    kernel get_rows_q_t kernel_get_rows_q<block_q1_0,    8, dequantize_q1_0>;
template [[host_name("kernel_get_rows_q2_0")]]    kernel get_rows_q_t kernel_get_rows_q<block_q2_0,    4, dequantize_q2_0>;
template [[host_name("kernel_get_rows_q4_0")]]    kernel get_rows_q_t kernel_get_rows_q<block_q4_0,    2, dequantize_q4_0>;
template [[host_name("kernel_get_rows_q4_1")]]    kernel get_rows_q_t kernel_get_rows_q<block_q4_1,    2, dequantize_q4_1>;
template [[host_name("kernel_get_rows_q5_0")]]    kernel get_rows_q_t kernel_get_rows_q<block_q5_0,    2, dequantize_q5_0>;
template [[host_name("kernel_get_rows_q5_1")]]    kernel get_rows_q_t kernel_get_rows_q<block_q5_1,    2, dequantize_q5_1>;
template [[host_name("kernel_get_rows_q8_0")]]    kernel get_rows_q_t kernel_get_rows_q<block_q8_0,    2, dequantize_q8_0>;
template [[host_name("kernel_get_rows_mxfp4")]]   kernel get_rows_q_t kernel_get_rows_q<block_mxfp4,   2, dequantize_mxfp4>;
template [[host_name("kernel_get_rows_q2_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q2_K,    QK_NL, dequantize_q2_K>;
template [[host_name("kernel_get_rows_q3_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q3_K,    QK_NL, dequantize_q3_K>;
template [[host_name("kernel_get_rows_q4_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q4_K,    QK_NL, dequantize_q4_K>;
template [[host_name("kernel_get_rows_q5_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q5_K,    QK_NL, dequantize_q5_K>;
template [[host_name("kernel_get_rows_q6_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q6_K,    QK_NL, dequantize_q6_K>;
template [[host_name("kernel_get_rows_iq2_xxs")]] kernel get_rows_q_t kernel_get_rows_q<block_iq2_xxs, QK_NL, dequantize_iq2_xxs>;
template [[host_name("kernel_get_rows_iq2_xs")]]  kernel get_rows_q_t kernel_get_rows_q<block_iq2_xs,  QK_NL, dequantize_iq2_xs>;
template [[host_name("kernel_get_rows_iq3_xxs")]] kernel get_rows_q_t kernel_get_rows_q<block_iq3_xxs, QK_NL, dequantize_iq3_xxs>;
template [[host_name("kernel_get_rows_iq3_s")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq3_s,   QK_NL, dequantize_iq3_s>;
template [[host_name("kernel_get_rows_iq2_s")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq2_s,   QK_NL, dequantize_iq2_s>;
template [[host_name("kernel_get_rows_iq1_s")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq1_s,   QK_NL, dequantize_iq1_s>;
template [[host_name("kernel_get_rows_iq1_m")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq1_m,   QK_NL, dequantize_iq1_m>;
template [[host_name("kernel_get_rows_iq4_nl")]]  kernel get_rows_q_t kernel_get_rows_q<block_iq4_nl,  2,     dequantize_iq4_nl>;
template [[host_name("kernel_get_rows_iq4_xs")]]  kernel get_rows_q_t kernel_get_rows_q<block_iq4_xs,  QK_NL, dequantize_iq4_xs>;
template [[host_name("kernel_get_rows_tq2_0")]]   kernel get_rows_q_t kernel_get_rows_q<block_tq2_0,   QK_NL, dequantize_tq2_0>;
template [[host_name("kernel_get_rows_turbo4")]] kernel get_rows_q_t kernel_get_rows_q<block_turbo4_0, 8, dequantize_turbo4_0>;
template [[host_name("kernel_get_rows_turbo2")]] kernel get_rows_q_t kernel_get_rows_q<block_turbo2_0, 2, dequantize_turbo2_0>;
template [[host_name("kernel_get_rows_turbo3")]] kernel get_rows_q_t kernel_get_rows_q<block_turbo3_0, 2, dequantize_turbo3_0>;

template<typename TS, typename TI, short QK, typename block_q, void (*quantize_func)(device const float *, device block_q &)>
kernel void kernel_set_rows_q(
        constant ggml_metal_kargs_set_rows & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;

    const int32_t i12 = i03%args.ne12;
    const int32_t i11 = i02%args.ne11;

    const int32_t i01 = tgpig.x*tptg.y + tiitg/tptg.x;
    if (i01 >= args.ne01) {
        return;
    }

    const int32_t i10 = i01;
    const TI      i1  = ((const device TI *) ((const device char *) src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

          device block_q * dst_row = (      device block_q *) ((      device char *) dst  +  i1*args.nb1  + i02*args.nb2  + i03*args.nb3);
    const device TS      * src_row = (const device TS      *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);

    for (int ind = tiitg%tptg.x; ind < args.nk0; ind += tptg.x) {
        quantize_func(src_row + QK*ind, dst_row[ind]);
    }
}

template<typename TS, typename TI, typename block_q, void (*quantize_func)(device const float *, device block_q &)>
kernel void kernel_set_rows_q32(
        constant ggml_metal_kargs_set_rows & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;

    const int32_t i12 = i03%args.ne12;
    const int32_t i11 = i02%args.ne11;

    const int32_t i01 = tgpig.x*tptg.y + tiitg/tptg.x;
    if (i01 >= args.ne01) {
        return;
    }

    const int32_t i10 = i01;
    const TI      i1  = ((const device TI *) ((const device char *) src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

          device block_q * dst_row = (      device block_q *) ((      device char *) dst  +  i1*args.nb1  + i02*args.nb2  + i03*args.nb3);
    const device TS      * src_row = (const device TS      *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);

    for (int ind = tiitg%tptg.x; ind < args.nk0; ind += tptg.x) {
        quantize_func(src_row + 32*ind, dst_row[ind]);
    }
}

template<typename TS, typename TI, typename TD>
kernel void kernel_set_rows_f(
        constant ggml_metal_kargs_set_rows & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;

    const int32_t i12 = i03%args.ne12;
    const int32_t i11 = i02%args.ne11;

    const int32_t i01 = tgpig.x*tptg.y + tiitg/tptg.x;
    if (i01 >= args.ne01) {
        return;
    }

    const int32_t i10 = i01;
    const TI      i1  = ((const device TI *) ((const device char *) src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

          device TD * dst_row = (      device TD *) ((      device char *) dst  +  i1*args.nb1  + i02*args.nb2  + i03*args.nb3);
    const device TS * src_row = (const device TS *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);

    for (int ind = tiitg%tptg.x; ind < args.nk0; ind += tptg.x) {
        dst_row[ind] = (TD) src_row[ind];
    }
}

typedef decltype(kernel_set_rows_f<float, int64_t, float>) set_rows_f_t;

template [[host_name("kernel_set_rows_f32_i64_f32")]]   kernel set_rows_f_t kernel_set_rows_f<float, int64_t, float>;
template [[host_name("kernel_set_rows_f32_i32_f32")]]   kernel set_rows_f_t kernel_set_rows_f<float, int32_t, float>;
template [[host_name("kernel_set_rows_f32_i64_f16")]]   kernel set_rows_f_t kernel_set_rows_f<float, int64_t, half>;
template [[host_name("kernel_set_rows_f32_i32_f16")]]   kernel set_rows_f_t kernel_set_rows_f<float, int32_t, half>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_set_rows_f32_i64_bf16")]]  kernel set_rows_f_t kernel_set_rows_f<float, int64_t, bfloat>;
template [[host_name("kernel_set_rows_f32_i32_bf16")]]  kernel set_rows_f_t kernel_set_rows_f<float, int32_t, bfloat>;
#endif

template [[host_name("kernel_set_rows_f16_i64_f16")]]   kernel set_rows_f_t kernel_set_rows_f<half, int64_t, half>;
template [[host_name("kernel_set_rows_f16_i32_f16")]]   kernel set_rows_f_t kernel_set_rows_f<half, int32_t, half>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_set_rows_bf16_i64_bf16")]] kernel set_rows_f_t kernel_set_rows_f<bfloat, int64_t, bfloat>;
template [[host_name("kernel_set_rows_bf16_i32_bf16")]] kernel set_rows_f_t kernel_set_rows_f<bfloat, int32_t, bfloat>;
#endif

typedef decltype(kernel_set_rows_q32<float, int64_t, block_q8_0, quantize_q8_0>) set_rows_q32_t;

template [[host_name("kernel_set_rows_f32_i64_q8_0")]]   kernel set_rows_q32_t kernel_set_rows_q32<float, int64_t, block_q8_0,   quantize_q8_0>;
template [[host_name("kernel_set_rows_f32_i32_q8_0")]]   kernel set_rows_q32_t kernel_set_rows_q32<float, int32_t, block_q8_0,   quantize_q8_0>;
template [[host_name("kernel_set_rows_f32_i64_q4_0")]]   kernel set_rows_q32_t kernel_set_rows_q32<float, int64_t, block_q4_0,   quantize_q4_0>;
template [[host_name("kernel_set_rows_f32_i32_q4_0")]]   kernel set_rows_q32_t kernel_set_rows_q32<float, int32_t, block_q4_0,   quantize_q4_0>;
template [[host_name("kernel_set_rows_f32_i64_q4_1")]]   kernel set_rows_q32_t kernel_set_rows_q32<float, int64_t, block_q4_1,   quantize_q4_1>;
template [[host_name("kernel_set_rows_f32_i32_q4_1")]]   kernel set_rows_q32_t kernel_set_rows_q32<float, int32_t, block_q4_1,   quantize_q4_1>;
template [[host_name("kernel_set_rows_f32_i64_q5_0")]]   kernel set_rows_q32_t kernel_set_rows_q32<float, int64_t, block_q5_0,   quantize_q5_0>;
template [[host_name("kernel_set_rows_f32_i32_q5_0")]]   kernel set_rows_q32_t kernel_set_rows_q32<float, int32_t, block_q5_0,   quantize_q5_0>;
template [[host_name("kernel_set_rows_f32_i64_q5_1")]]   kernel set_rows_q32_t kernel_set_rows_q32<float, int64_t, block_q5_1,   quantize_q5_1>;
template [[host_name("kernel_set_rows_f32_i32_q5_1")]]   kernel set_rows_q32_t kernel_set_rows_q32<float, int32_t, block_q5_1,   quantize_q5_1>;
template [[host_name("kernel_set_rows_f32_i64_iq4_nl")]] kernel set_rows_q32_t kernel_set_rows_q32<float, int64_t, block_iq4_nl, quantize_iq4_nl>;
template [[host_name("kernel_set_rows_f32_i32_iq4_nl")]] kernel set_rows_q32_t kernel_set_rows_q32<float, int32_t, block_iq4_nl, quantize_iq4_nl>;

typedef decltype(kernel_set_rows_q<float, int64_t, QK_K, block_tq2_0, quantize_tq2_0>) set_rows_qK_t;

template [[host_name("kernel_set_rows_f32_i64_tq2_0")]]  kernel set_rows_qK_t kernel_set_rows_q<float, int64_t, QK_K, block_tq2_0, quantize_tq2_0>;
template [[host_name("kernel_set_rows_f32_i32_tq2_0")]]  kernel set_rows_qK_t kernel_set_rows_q<float, int32_t, QK_K, block_tq2_0, quantize_tq2_0>;

template<typename TS, typename TI>
kernel void kernel_set_rows_turbo4(
        constant ggml_metal_kargs_set_rows & args,
        device const void * src0,
        device const void * src1,
        device       void * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;

    const int32_t i12 = i03%args.ne12;
    const int32_t i11 = i02%args.ne11;

    const int32_t i01 = tgpig.x*tptg.y + tiitg/tptg.x;
    if (i01 >= args.ne01) {
        return;
    }

    const int32_t i10 = i01;
    const TI      i1  = ((const device TI *) ((const device char *) src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

          device block_turbo4_0 * dst_row = (      device block_turbo4_0 *) ((      device char *) dst  +  i1*args.nb1  + i02*args.nb2  + i03*args.nb3);
    const device TS             * src_row = (const device TS             *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);

    for (int ind = tiitg%tptg.x; ind < args.nk0; ind += tptg.x) {
        device block_turbo4_0 & blk = dst_row[ind];
        float x[128];
        float norm_sq = 0.0f;
        for (short j = 0; j < 128; ++j) {
            float v = (float) src_row[128*ind + j];
            x[j] = v;
            norm_sq += v * v;
        }
        float norm = sqrt(norm_sq);
        float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;
        for (short j = 0; j < 128; ++j) {
            x[j] *= inv_norm;
        }
        turbo_rotate_forward(x, d_turbo_wht_signs1_fattn, d_turbo_wht_signs2_fattn);

        float recon_sq = 0.0f;
        for (short j = 0; j < 128; j += 2) {
            uchar idx0 = turbo_find_nearest_4bit(x[j]);
            uchar idx1 = turbo_find_nearest_4bit(x[j + 1]);
            blk.qs[j / 2] = (idx1 << 4) | idx0;
            float r0 = d_turbo_centroids_4bit_fattn[idx0];
            float r1 = d_turbo_centroids_4bit_fattn[idx1];
            recon_sq += r0 * r0 + r1 * r1;
        }
        float recon_norm = sqrt(recon_sq);
        float corrected = (recon_norm > 1e-10f) ? norm / recon_norm : norm;
        blk.norm = (half) corrected;
    }
}

template<typename TS, typename TI>
kernel void kernel_set_rows_turbo2(
        constant ggml_metal_kargs_set_rows & args,
        device const void * src0,
        device const void * src1,
        device       void * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;

    const int32_t i12 = i03%args.ne12;
    const int32_t i11 = i02%args.ne11;

    const int32_t i01 = tgpig.x*tptg.y + tiitg/tptg.x;
    if (i01 >= args.ne01) {
        return;
    }

    const int32_t i10 = i01;
    const TI      i1  = ((const device TI *) ((const device char *) src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

          device block_turbo2_0 * dst_row = (      device block_turbo2_0 *) ((      device char *) dst  +  i1*args.nb1  + i02*args.nb2  + i03*args.nb3);
    const device TS             * src_row = (const device TS             *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);

    const int n_groups = args.nk0 / 4;
    for (int grp = tiitg%tptg.x; grp < n_groups; grp += tptg.x) {
        float x[128];
        float norm_sq = 0.0f;
        for (short j = 0; j < 128; ++j) {
            float v = (float) src_row[128*grp + j];
            x[j] = v;
            norm_sq += v * v;
        }
        float grp_norm = sqrt(norm_sq);
        float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
        for (short j = 0; j < 128; ++j) {
            x[j] *= inv_norm;
        }
        turbo_rotate_forward(x, d_turbo_wht_signs1_fattn, d_turbo_wht_signs2_fattn);

        float recon_norm_sq = 0.0f;
        for (short b = 0; b < 4; ++b) {
            device block_turbo2_0 & blk = dst_row[4*grp + b];
            const short off = b * 32;
            for (short j = 0; j < 8; ++j) blk.qs[j] = 0;
            for (short j = 0; j < 32; ++j) {
                uchar idx = turbo_find_nearest_2bit(x[off + j]);
                blk.qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
                float c = d_turbo_centroids_2bit_fattn[idx];
                recon_norm_sq += c * c;
            }
        }
        float recon_norm = sqrt(recon_norm_sq);
        float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        for (short b = 0; b < 4; ++b) {
            dst_row[4*grp + b].norm = (half) corrected_norm;
        }
    }
}

template<typename TS, typename TI>
kernel void kernel_set_rows_turbo3(
        constant ggml_metal_kargs_set_rows & args,
        device const void * src0,
        device const void * src1,
        device       void * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;

    const int32_t i12 = i03%args.ne12;
    const int32_t i11 = i02%args.ne11;

    const int32_t i01 = tgpig.x*tptg.y + tiitg/tptg.x;
    if (i01 >= args.ne01) {
        return;
    }

    const int32_t i10 = i01;
    const TI      i1  = ((const device TI *) ((const device char *) src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

          device block_turbo3_0 * dst_row = (      device block_turbo3_0 *) ((      device char *) dst  +  i1*args.nb1  + i02*args.nb2  + i03*args.nb3);
    const device TS             * src_row = (const device TS             *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);

    const int n_groups = args.nk0 / 4;
    for (int grp = tiitg%tptg.x; grp < n_groups; grp += tptg.x) {
        float x[128];
        float norm_sq = 0.0f;
        for (short j = 0; j < 128; ++j) {
            float v = (float) src_row[128*grp + j];
            x[j] = v;
            norm_sq += v * v;
        }
        float grp_norm = sqrt(norm_sq);
        float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
        for (short j = 0; j < 128; ++j) {
            x[j] *= inv_norm;
        }
        turbo_rotate_forward(x, d_turbo_wht_signs1_fattn, d_turbo_wht_signs2_fattn);

        float recon_norm_sq = 0.0f;
        for (short b = 0; b < 4; ++b) {
            device block_turbo3_0 & blk = dst_row[4*grp + b];
            const short off = b * 32;
            for (short j = 0; j < 8; ++j) blk.qs[j] = 0;
            for (short j = 0; j < 4; ++j) blk.signs[j] = 0;
            for (short j = 0; j < 32; ++j) {
                uchar idx = turbo_find_nearest_3bit(x[off + j]);
                blk.qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
                blk.signs[j / 8] |= ((idx >> 2) & 0x1) << (j % 8);
                float c = d_turbo_centroids_3bit_fattn[idx];
                recon_norm_sq += c * c;
            }
        }
        float recon_norm = sqrt(recon_norm_sq);
        float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        for (short b = 0; b < 4; ++b) {
            dst_row[4*grp + b].norm = (half) corrected_norm;
        }
    }
}

typedef decltype(kernel_set_rows_turbo4<float, int64_t>) set_rows_t4_f32_i64_t;
typedef decltype(kernel_set_rows_turbo4<float, int32_t>) set_rows_t4_f32_i32_t;
typedef decltype(kernel_set_rows_turbo4<half,  int64_t>) set_rows_t4_f16_i64_t;
typedef decltype(kernel_set_rows_turbo4<half,  int32_t>) set_rows_t4_f16_i32_t;

template [[host_name("kernel_set_rows_f32_i64_turbo4")]] kernel set_rows_t4_f32_i64_t kernel_set_rows_turbo4<float, int64_t>;
template [[host_name("kernel_set_rows_f32_i32_turbo4")]] kernel set_rows_t4_f32_i32_t kernel_set_rows_turbo4<float, int32_t>;
template [[host_name("kernel_set_rows_f16_i64_turbo4")]] kernel set_rows_t4_f16_i64_t kernel_set_rows_turbo4<half,  int64_t>;
template [[host_name("kernel_set_rows_f16_i32_turbo4")]] kernel set_rows_t4_f16_i32_t kernel_set_rows_turbo4<half,  int32_t>;

typedef decltype(kernel_set_rows_turbo2<float, int64_t>) set_rows_t2_f32_i64_t;
typedef decltype(kernel_set_rows_turbo2<float, int32_t>) set_rows_t2_f32_i32_t;
typedef decltype(kernel_set_rows_turbo2<half,  int64_t>) set_rows_t2_f16_i64_t;
typedef decltype(kernel_set_rows_turbo2<half,  int32_t>) set_rows_t2_f16_i32_t;

template [[host_name("kernel_set_rows_f32_i64_turbo2")]] kernel set_rows_t2_f32_i64_t kernel_set_rows_turbo2<float, int64_t>;
template [[host_name("kernel_set_rows_f32_i32_turbo2")]] kernel set_rows_t2_f32_i32_t kernel_set_rows_turbo2<float, int32_t>;
template [[host_name("kernel_set_rows_f16_i64_turbo2")]] kernel set_rows_t2_f16_i64_t kernel_set_rows_turbo2<half,  int64_t>;
template [[host_name("kernel_set_rows_f16_i32_turbo2")]] kernel set_rows_t2_f16_i32_t kernel_set_rows_turbo2<half,  int32_t>;

typedef decltype(kernel_set_rows_turbo3<float, int64_t>) set_rows_t3_f32_i64_t;
typedef decltype(kernel_set_rows_turbo3<float, int32_t>) set_rows_t3_f32_i32_t;
typedef decltype(kernel_set_rows_turbo3<half,  int64_t>) set_rows_t3_f16_i64_t;
typedef decltype(kernel_set_rows_turbo3<half,  int32_t>) set_rows_t3_f16_i32_t;

template [[host_name("kernel_set_rows_f32_i64_turbo3")]] kernel set_rows_t3_f32_i64_t kernel_set_rows_turbo3<float, int64_t>;
template [[host_name("kernel_set_rows_f32_i32_turbo3")]] kernel set_rows_t3_f32_i32_t kernel_set_rows_turbo3<float, int32_t>;
template [[host_name("kernel_set_rows_f16_i64_turbo3")]] kernel set_rows_t3_f16_i64_t kernel_set_rows_turbo3<half,  int64_t>;
template [[host_name("kernel_set_rows_f16_i32_turbo3")]] kernel set_rows_t3_f16_i32_t kernel_set_rows_turbo3<half,  int32_t>;

