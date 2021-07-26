#pragma once
#include <iostream>
#include "dtu/tops/tops.h"
#include "oneflow/core/common/data_type.h"

inline topsDataType oneflowDataType2TopsDType(oneflow::DataType type)
{
    switch (type)
    {
    case oneflow::DataType::kFloat:
        return TOPS_DATA_FLOAT;
    case oneflow::DataType::kDouble:
        return TOPS_DATA_DOUBLE;
    case oneflow::DataType::kFloat16:
        return TOPS_DATA_HALF;
    case oneflow::DataType::kInt8:
        return TOPS_DATA_INT8;
    case oneflow::DataType::kInt32:
        return TOPS_DATA_INT32;
    case oneflow::DataType::kUInt8:
        return TOPS_DATA_UINT8;
    default:
        assert(false);
    }
}

template<class T>
topsMemory_t scalar2tensor(topsContext_t context, topsDataType dtype, T scalar, topsTensorNdDescriptor_t tensorDesc)
{
    // device memory for input scalar
    topsMemory_t scalarMem = nullptr;
    topsMalloc(context, &scalarMem, sizeof(scalar));
    topsMemcpyHostToDevice(context, &scalar, scalarMem);
    // scalar desc
    int scalarDescDim[0];
    topsTensorNdDescriptor_t scalarDesc = nullptr;
    topsCreateTensorNdDescriptor(&scalarDesc);
    topsSetTensorNdDescriptor(scalarDesc, dtype, 0, scalarDescDim);

    // run broadcast
    topsMemory_t resultMem = nullptr;
    topsBroadcast(context, scalarDesc, scalarMem, scalarDescDim, tensorDesc, &resultMem);
    
    topsFree(context, scalarMem);
    topsDestroyTensorNdDescriptor(scalarDesc);
    return resultMem;
}

template<class T>
topsMemory_t scalarMulTensor(topsContext_t context, topsDataType dtype, T scalar, topsTensorNdDescriptor_t tensorDesc, topsMemory_t tensorMem)
{
    topsMemory_t broadcastMem = scalar2tensor<T>(context, dtype, scalar, tensorDesc);
    // run elementwise binary
    topsMemory_t resultMem = nullptr;
    topsBinaryOp(context, TOPS_BINARY_OP_MUL, 
        tensorDesc, tensorMem, tensorDesc, broadcastMem, tensorDesc, &resultMem);
    topsFree(context, broadcastMem);
    return resultMem;
}