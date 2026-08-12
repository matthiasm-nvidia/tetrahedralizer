// SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#pragma once

#include "tetrahedralizer/Vec.h"

#include <cstdio>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define THREADS_PER_BLOCK 256
#define CUDA_UTILS_DEBUG 0

#define CUDA_LINEAR_THREAD_IDX (threadIdx.x + blockIdx.x * blockDim.x)

#define CUDA_THREAD_GUARD(name, maxExclusive)                                                                          \
    int name = CUDA_LINEAR_THREAD_IDX;                                                                                 \
    if ((name) >= (maxExclusive))                                                                                      \
        return;

#define CUDA_LAUNCH(kernel, numThreads, ...)                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        const int _mt_numThreads = static_cast<int>(numThreads);                                                       \
        if (_mt_numThreads > 0)                                                                                        \
        {                                                                                                              \
            const int _mt_grid = (_mt_numThreads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;                         \
            kernel<<<_mt_grid, THREADS_PER_BLOCK>>>(__VA_ARGS__);                                                      \
            cudaCheck(cudaGetLastError());                                                                             \
        }                                                                                                              \
    } while (0)

#define cudaCheck(x)                                                                                                   \
    {                                                                                                                  \
        cudaAssert((x), #x, __FILE__, __LINE__);                                                                       \
    }

inline std::string formatCudaVersion(int version)
{
    if (version <= 0)
        return "unknown";
    return std::to_string(version / 1000) + "." + std::to_string((version % 1000) / 10);
}

inline void cudaAssert(cudaError_t code, const char* call, const char* file, int line)
{
    if (code != cudaSuccess)
    {
        int driverVersion = 0;
        int runtimeVersion = 0;
        const cudaError_t driverVersionError = cudaDriverGetVersion(&driverVersion);
        const cudaError_t runtimeVersionError = cudaRuntimeGetVersion(&runtimeVersion);

        std::string errorMsg =
            std::string("CUDA error: ") + cudaGetErrorString(code) + " at " + file + ":" + std::to_string(line);
        errorMsg += " (call=";
        errorMsg += call ? call : "unknown";
        errorMsg += ")";

        if (code == cudaErrorInsufficientDriver)
        {
            errorMsg += " (driver=";
            errorMsg += driverVersionError == cudaSuccess ? formatCudaVersion(driverVersion) : "unknown";
            errorMsg += ", runtime=";
            errorMsg += runtimeVersionError == cudaSuccess ? formatCudaVersion(runtimeVersion) : "unknown";
            errorMsg += ")";
        }

        fprintf(stderr, "%s\n", errorMsg.c_str());
        throw std::runtime_error(errorMsg);
    }
}

cudaTextureObject_t CreateTexture(const tetrahedralizer::Vec4* buffer, int sizeInBytes);
void DestroyTexture(cudaTextureObject_t tex);
void DeviceAlloc(void** ptr, size_t size);
void DeviceFree(void* ptr);

template <typename T>
struct DeviceBuffer
{
    DeviceBuffer() : buffer(nullptr), size(0), capacity(0)
    {
    }

    void free()
    {
        if (buffer)
            cudaCheck(cudaFree(buffer));
        capacity = 0;
        size = 0;
        buffer = nullptr;
    }

    void clear()
    {
        size = 0;
    }

    void swap(DeviceBuffer& other) noexcept
    {
        std::swap(buffer, other.buffer);
        std::swap(size, other.size);
        std::swap(capacity, other.capacity);
    }

    void resize(size_t newSize, bool preserveData = true)
    {
        if (newSize <= capacity)
        {
            size = newSize;
            return;
        }

        T* oldBuffer = buffer;
        size_t oldSize = size;

        cudaCheck(cudaMalloc((void**)&buffer, newSize * sizeof(T)));
        cudaCheck(cudaMemset(buffer, 0, newSize * sizeof(T)));
        size = capacity = newSize;

        if (oldBuffer)
        {
            if (preserveData)
                cudaCheck(cudaMemcpy(buffer, oldBuffer, oldSize * sizeof(T), cudaMemcpyDeviceToDevice));
            cudaCheck(cudaFree(oldBuffer));
        }
    }

    void setZero()
    {
        if (buffer)
            cudaCheck(cudaMemset(buffer, 0, size * sizeof(T)));
    }

    void set(const std::vector<T>& hostBuffer, int num = 0)
    {
        if (num == 0)
            num = (int)hostBuffer.size();
        if (num == 0)
            return;
        resize(num);
        cudaCheck(cudaMemcpy(buffer, hostBuffer.data(), num * sizeof(T), cudaMemcpyHostToDevice));
    }

    void set(const T* hostBuffer, size_t num, bool resizeBuffer = true)
    {
        if (num == 0)
        {
            size = 0;
            return;
        }
        if (resizeBuffer)
            resize(num);
        else
            num = std::min(num, size);
        cudaCheck(cudaMemcpy(buffer, hostBuffer, num * sizeof(T), cudaMemcpyHostToDevice));
    }

    void get(std::vector<T>& hostBuffer, size_t num = 0)
    {
        if (num == 0)
            num = size;
        hostBuffer.resize(num);
        if (num == 0)
            return;
        cudaCheck(cudaMemcpy(hostBuffer.data(), buffer, num * sizeof(T), cudaMemcpyDeviceToHost));
    }

    size_t allocationSize() const
    {
        return capacity * sizeof(T);
    }

    __host__ __device__ T& operator[](size_t i)
    {
        return buffer[i];
    }

    T* buffer;
    size_t size;
    size_t capacity;
};
