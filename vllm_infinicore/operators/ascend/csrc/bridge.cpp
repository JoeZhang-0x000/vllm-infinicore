// The caller owns the active device and stream. This bridge never initializes,
// selects, resets, or finalizes an Ascend device/runtime.
#include "infiniop/devices/ascend/ascend_handle.h"
#include <new>

extern "C" const char *vllmInfinicoreRevision() { return INFINICORE_REVISION; }
extern "C" int vllmInfinicoreBridgeABI() { return 1; }
extern "C" int vllmInfinicoreCreateAscendHandle(InfiniopHandle **handle, int device) {
    if (!handle) return INFINI_STATUS_NULL_POINTER;
    try { return device::ascend::Handle::create(handle, device); }
    catch (...) { return INFINI_STATUS_INTERNAL_ERROR; }
}
extern "C" void vllmInfinicoreDestroyAscendHandle(InfiniopHandle *handle) {
    delete static_cast<device::ascend::Handle *>(handle);
}

// At the pinned revision, the Embedding C API omits ASCEND in its destroy
// switch although create/calculate support it. Destroy the upstream concrete
// descriptor here; do not modify the pinned source or leak its ACL workspace.
#include "infiniop/ops/embedding/ascend/embedding_ascend.h"
extern "C" int vllmInfinicoreDestroyEmbeddingDescriptor(InfiniopDescriptor *desc) {
    if (!desc) return INFINI_STATUS_NULL_POINTER;
    if (desc->device_type != INFINI_DEVICE_ASCEND)
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    delete static_cast<op::embedding::ascend::Descriptor *>(desc);
    return INFINI_STATUS_SUCCESS;
}
