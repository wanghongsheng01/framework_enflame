# Enflame

[Enflame 仓库](https://github.com/wanghongsheng01/Enflame)<br>

 Enflame supports Oneflow deep learning framework<br>

在 GPU/topsDevice 卡上计算的执行逻辑：<br>

1. init device，初始化 topsDevice<br>
   ```.cc
   topsContext_t context;
   int clusters[] = {0};
   topsDeviceCreate(&context, 0, 1, clusters);
   ```
    
2. inputs<br>
   ```.cc
   const int COUNT = 9;
   float InputDataA[COUNT] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
   float InputDataB[COUNT] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
   ```
3. memory for inputs <br>
   在 topsDevice 上为全部的输入申请 `topsMalloc` 内存(tops-Memory-allocate)<br>
   ```.cc
   topsMemory_t input_mem = nullptr;
   topsMalloc(context, &input_mem, sizeof(InputDataA) + sizeof(InputDataB));
   ```
5. copy input data from host<br>
   把 input data 从 cpu host 机器上拷贝到 topsDevice 上内存里（topsMalloc 提前申请了 input data 总的内存）<br>
   ```.cc
   topsMemory_t mem1 = nullptr;
   topsSubMemory(input_mem, 0, sizeof(InputDataA), &mem1);
   topsMemcpyHostToDevice(context, InputDataA, mem1);
   topsMemory_t mem2 = nullptr;
   topsSubMemory(input_mem, sizeof(InputDataA), sizeof(InputDataB), &mem2);
   topsMemcpyHostToDevice(context, InputDataB, mem2);
   ```
 6. construct operands for op<br>
    为 op 构造操作数 tensor<br>
    
    构造加数 tensor a <br>
    ```.cc
    int aDescDim[] = {3, 3};
    topsTensorNdDescriptor_t aDesc;
    topsCreateTensorNdDescriptor(&aDesc);
    topsSetTensorNdDescriptor(aDesc, TOPS_DATA_FLOAT, 2, aDescDim);
    ```
    构造加数 tensor b<br>
    ```.cc
    int bDescDim[] = {3, 3};
    topsTensorNdDescriptor_t bDesc;
    topsCreateTensorNdDescriptor(&bDesc);
    topsSetTensorNdDescriptor(bDesc, TOPS_DATA_FLOAT, 2, bDescDim);
    ```
    构造返回结果 tensor c<br>
    ```.cc
    int cDescDim[] = {3, 3};
    topsTensorNdDescriptor_t cDesc;
    topsCreateTensorNdDescriptor(&cDesc);
    topsSetTensorNdDescriptor(cDesc, TOPS_DATA_FLOAT, 2, cDescDim);
    ```
7. run op<br>
   运行计算 op，注意 topsBinaryOp 参数除了输入输出 tensor，还有输入 tensor 已分配 <br>
   `topsMemory_t mem1 = nullptr;topsSubMemory(input_mem, 0, sizeof(InputDataA), &mem1);`<br>
   的内存 mem1/mem2<br>
    
   ```.cc
   topsMemory_t mem3;
   topsBinaryOp(context, TOPS_BINARY_OP_ADD, aDesc, mem1, bDesc, mem2, cDesc, (&mem3));
   ```
8. copy result back to host<br>
   将 topsDevice 上计算所得的结果拷贝回 host 主机上<br>
   ```.cc
   float OutputData[COUNT] = {0};
   topsMemcpyDeviceToHost(context, mem3, OutputData);
   ```
9. release resources<br>
   释放 topsDevice 卡上的资源<br>
   ```.cc
   topsFree(context, input_mem);
   topsFree(context, mem1);
   topsFree(context, mem2);
   topsFree(context, mem3);
   topsDestroyTensorNdDescriptor(aDesc);
   topsDestroyTensorNdDescriptor(bDesc);
   topsDestroyTensorNdDescriptor(cDesc);
   topsDeviceDestroy(context);
   ```
    
打印 host 输出结果<br>
```.cc
std::cout << "Add Result: \n";
for (int i = 0; i < COUNT; ++i)
    std::cout << OutputData[i] << ", ";
std::cout << std::endl;
```
    
tops_add.cc<br>
```tops_add.cc

#include <iostream>
#include <memory>
#include <thread>
#include "dtu/tops/tops.h"


void topsElementWiseAdd() 
{
    // init device
    topsContext_t context;
    int clusters[] = {0};
    topsDeviceCreate(&context, 0, 1, clusters);

    // inputs
    const int COUNT = 9;
    float InputDataA[COUNT] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float InputDataB[COUNT] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    // memory for inputs
    topsMemory_t input_mem = nullptr;
    topsMalloc(context, &input_mem, sizeof(InputDataA) + sizeof(InputDataB));
    // copy input data from host
    topsMemory_t mem1 = nullptr;
    topsSubMemory(input_mem, 0, sizeof(InputDataA), &mem1);
    topsMemcpyHostToDevice(context, InputDataA, mem1);
    topsMemory_t mem2 = nullptr;
    topsSubMemory(input_mem, sizeof(InputDataA), sizeof(InputDataB), &mem2);
    topsMemcpyHostToDevice(context, InputDataB, mem2);

    // construct operands for op
    int aDescDim[] = {3, 3};
    topsTensorNdDescriptor_t aDesc;
    topsCreateTensorNdDescriptor(&aDesc);
    topsSetTensorNdDescriptor(aDesc, TOPS_DATA_FLOAT, 2, aDescDim);
    int bDescDim[] = {3, 3};
    topsTensorNdDescriptor_t bDesc;
    topsCreateTensorNdDescriptor(&bDesc);
    topsSetTensorNdDescriptor(bDesc, TOPS_DATA_FLOAT, 2, bDescDim);
    int cDescDim[] = {3, 3};
    topsTensorNdDescriptor_t cDesc;
    topsCreateTensorNdDescriptor(&cDesc);
    topsSetTensorNdDescriptor(cDesc, TOPS_DATA_FLOAT, 2, cDescDim);
    
    // run op
    topsMemory_t mem3;
    topsBinaryOp(context, TOPS_BINARY_OP_ADD, 
        aDesc, mem1, bDesc, mem2, cDesc, (&mem3));

    // copy result back to host
    float OutputData[COUNT] = {0};
    topsMemcpyDeviceToHost(context, mem3, OutputData);

    // release resources
    topsFree(context, input_mem);
    topsFree(context, mem1);
    topsFree(context, mem2);
    topsFree(context, mem3);
    topsDestroyTensorNdDescriptor(aDesc);
    topsDestroyTensorNdDescriptor(bDesc);
    topsDestroyTensorNdDescriptor(cDesc);
    topsDeviceDestroy(context);

    // print result
    std::cout << "Add Result: \n";
    for (int i = 0; i < COUNT; ++i)
        std::cout << OutputData[i] << ", ";
    std::cout << std::endl;
}

void topsElementWiseAdd2() 
{
    // init device
    topsContext_t context;
    int clusters[] = {0};
    topsDeviceCreate(&context, 0, 1, clusters);
    // data
    const int COUNT = 9;
    float InputData[COUNT] = {0};
    float OutputData[COUNT] = {0};
    // copy to device
    topsMemory_t mem1;
    topsMalloc(context, &mem1, sizeof(InputData));
    topsMemory_t mem2;
    topsMalloc(context, &mem2, sizeof(OutputData));
    topsMemcpyHostToDevice(context, OutputData, mem2);
    // construct operands for op
    topsTensorNdDescriptor_t desc;
    topsCreateTensorNdDescriptor(&desc);
    int dims[] = {3, 3};
    topsSetTensorNdDescriptor(desc, TOPS_DATA_FLOAT, 2, dims);
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < COUNT; ++j)
            InputData[j] = i;
        topsMemcpyHostToDevice(context, InputData, mem1);
        // run op
        topsBinaryOp(context, TOPS_BINARY_OP_ADD, 
            desc, mem1, desc, mem2, desc, &mem2);
        std::cout << "mem2: " << (uint64_t)mem2 << std::endl;
        topsMemcpyDeviceToHost(context, mem2, OutputData);
        // print result
        std::cout << "Add Result: \n";
        for (int j = 0; j < COUNT; ++j)
            std::cout << OutputData[j] << ", ";
        std::cout << std::endl;
    }

    // release resources
    topsFree(context, mem1);
    topsFree(context, mem2);
    topsDestroyTensorNdDescriptor(desc);
    topsDeviceDestroy(context);
}

void topsElementWiseAdd3() 
{
    // init device
    topsContext_t context;
    int clusters[] = {2};
    topsDeviceCreate(&context, 0, 1, clusters);
    // data
    const int COUNT = 9;
    float InputData[COUNT] = {0};
    float OutputData[COUNT] = {0};
    // copy to device
    topsMemory_t mem1;
    topsMalloc(context, &mem1, sizeof(InputData));
    std::cout << "mem1: " << std::hex << (long unsigned int)mem1 << std::endl;
    void* ptr1 = mem1;
    std::cout << "ptr1: " << std::hex << (long unsigned int)ptr1 << std::endl;
    
    // topsFree(context, mem1);
    // topsDeviceDestroy(context);
    // return;

    topsMemory_t mem2;
    topsMalloc(context, &mem2, sizeof(OutputData));
    topsMemcpyHostToDevice(context, OutputData, mem2);
    void* ptr2 = mem2;
    // construct operands for op
    topsTensorNdDescriptor_t desc;
    topsCreateTensorNdDescriptor(&desc);
    int dims[] = {3, 3};
    topsSetTensorNdDescriptor(desc, TOPS_DATA_FLOAT, 2, dims);
    for (int i = 0; i < 3; ++i)
    {
        auto local_mem1 = static_cast<topsMemory_t>(ptr1);
        auto local_mem2 = static_cast<topsMemory_t>(ptr2);
        for (int j = 0; j < COUNT; ++j)
            InputData[j] = i;
        topsMemcpyHostToDevice(context, InputData, local_mem1);
        // run op
        topsBinaryOp(context, TOPS_BINARY_OP_ADD, 
            desc, local_mem1, 
            desc, local_mem2, 
            desc, &local_mem2);
        ptr2 = local_mem2;
//        topsMemcpyDeviceToHost(context, local_mem2, OutputData);
        // print result
        std::cout << "Add Result: \n";
        for (int j = 0; j < COUNT; ++j)
            std::cout << OutputData[j] << ", ";
        std::cout << std::endl;
    }

    // release resources
    topsFree(context, mem1);
    topsFree(context, mem2);
    topsDestroyTensorNdDescriptor(desc);
    topsDeviceDestroy(context);
}

void compute1(topsContext_t context)
{
    // inputs
    const int COUNT = 9;
    float InputDataA[COUNT] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float InputDataB[COUNT] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    // memory for inputs
    topsMemory_t input_mem = nullptr;
    topsMalloc(context, &input_mem, sizeof(InputDataA) + sizeof(InputDataB));
    // copy input data from host
    topsMemory_t mem1 = nullptr;
    topsSubMemory(input_mem, 0, sizeof(InputDataA), &mem1);
    topsMemcpyHostToDevice(context, InputDataA, mem1);
    topsMemory_t mem2 = nullptr;
    topsSubMemory(input_mem, sizeof(InputDataA), sizeof(InputDataB), &mem2);
    topsMemcpyHostToDevice(context, InputDataB, mem2);

    // construct operands for op
    int aDescDim[] = {3, 3};
    topsTensorNdDescriptor_t aDesc;
    topsCreateTensorNdDescriptor(&aDesc);
    topsSetTensorNdDescriptor(aDesc, TOPS_DATA_FLOAT, 2, aDescDim);
    int bDescDim[] = {3, 3};
    topsTensorNdDescriptor_t bDesc;
    topsCreateTensorNdDescriptor(&bDesc);
    topsSetTensorNdDescriptor(bDesc, TOPS_DATA_FLOAT, 2, bDescDim);
    int cDescDim[] = {3, 3};
    topsTensorNdDescriptor_t cDesc;
    topsCreateTensorNdDescriptor(&cDesc);
    topsSetTensorNdDescriptor(cDesc, TOPS_DATA_FLOAT, 2, cDescDim);
    
    // run op
    topsMemory_t mem3;
    topsBinaryOp(context, TOPS_BINARY_OP_ADD, 
        aDesc, mem1, bDesc, mem2, cDesc, (&mem3));

    // copy result back to host
    float OutputData[COUNT] = {0};
    topsMemcpyDeviceToHost(context, mem3, OutputData);
    // print result
    std::cout << "Add Result: \n";
    for (int i = 0; i < COUNT; ++i)
        std::cout << OutputData[i] << ", ";
    std::cout << std::endl;

    // release resources
    topsFree(context, input_mem);
    topsFree(context, mem1);
    topsFree(context, mem2);
    topsFree(context, mem3);
    topsDestroyTensorNdDescriptor(aDesc);
    topsDestroyTensorNdDescriptor(bDesc);
    topsDestroyTensorNdDescriptor(cDesc);
}

void topsElementWiseAdd4() 
{
    // init device
    topsContext_t context;
    int clusters[] = {1};
    topsDeviceCreate(&context, 0, 1, clusters);

    std::thread compute_thread(compute1, context);
    compute_thread.join();

    topsDeviceDestroy(context);
}

void compute2(topsContext_t context, topsMemory_t mem1, topsMemory_t mem2, topsMemory_t mem3)
{
    // construct operands for op
    int aDescDim[] = {3, 3};
    topsTensorNdDescriptor_t aDesc;
    topsCreateTensorNdDescriptor(&aDesc);
    topsSetTensorNdDescriptor(aDesc, TOPS_DATA_FLOAT, 2, aDescDim);
    int bDescDim[] = {3, 3};
    topsTensorNdDescriptor_t bDesc;
    topsCreateTensorNdDescriptor(&bDesc);
    topsSetTensorNdDescriptor(bDesc, TOPS_DATA_FLOAT, 2, bDescDim);
    int cDescDim[] = {3, 3};
    topsTensorNdDescriptor_t cDesc;
    topsCreateTensorNdDescriptor(&cDesc);
    topsSetTensorNdDescriptor(cDesc, TOPS_DATA_FLOAT, 2, cDescDim);
    
    // run op
    topsBinaryOp(context, TOPS_BINARY_OP_ADD, 
        aDesc, mem1, bDesc, mem2, cDesc, (&mem3));

    topsDestroyTensorNdDescriptor(aDesc);
    topsDestroyTensorNdDescriptor(bDesc);
    topsDestroyTensorNdDescriptor(cDesc);
}

void topsElementWiseAdd5() 
{
    // init device
    topsContext_t context;
    int clusters[] = {0};
    topsDeviceCreate(&context, 0, 1, clusters);

    // inputs
    const int COUNT = 9;
    float InputDataA[COUNT] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float InputDataB[COUNT] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    // memory for inputs
    topsMemory_t input_mem = nullptr;
    topsMalloc(context, &input_mem, sizeof(InputDataA) + sizeof(InputDataB));
    // copy input data from host
    topsMemory_t mem1 = nullptr;
    topsSubMemory(input_mem, 0, sizeof(InputDataA), &mem1);
    topsMemcpyHostToDevice(context, InputDataA, mem1);
    topsMemory_t mem2 = nullptr;
    topsSubMemory(input_mem, sizeof(InputDataA), sizeof(InputDataB), &mem2);
    topsMemcpyHostToDevice(context, InputDataB, mem2);

    topsMemory_t mem3 = nullptr;

    std::thread compute_thread(compute2, context, mem1, mem2, mem3);
    compute_thread.join();

    // copy result back to host
    float OutputData[COUNT] = {0};
    topsMemcpyDeviceToHost(context, mem3, OutputData);
    // print result
    std::cout << "Add Result: \n";
    for (int i = 0; i < COUNT; ++i)
        std::cout << OutputData[i] << ", ";
    std::cout << std::endl;

    // release resources
    topsFree(context, input_mem);
    topsFree(context, mem1);
    topsFree(context, mem2);
    topsFree(context, mem3);

    topsDeviceDestroy(context);
}

int main() 
{
    topsElementWiseAdd2();
    return 0;
}

```
