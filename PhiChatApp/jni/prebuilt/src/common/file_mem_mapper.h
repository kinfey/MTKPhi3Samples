#include "common/logging.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

class FileMemMapper { // Read-only mmap
public:
    explicit FileMemMapper(const std::string& path) {
        // Get fd
        mFd = open(path.c_str(), O_RDONLY);
        if (mFd == -1) {
            LOG(ERROR) << "Open file fail: " << path;
            return;
        }

        // Get size
        struct stat sb;
        if (fstat(mFd, &sb) == -1) {
            LOG(ERROR) << "fstat fail";
            return;
        }
        mSize = sb.st_size;

        // Map file data to memory
        mBuffer = mmap(NULL, mSize, PROT_READ, MAP_SHARED, mFd, 0);
        if (mBuffer == MAP_FAILED) {
            LOG(ERROR) << "mmap fail";
            return;
        }

        LOG(DEBUG) << "FileMemMapper: Mapped to "
                   << "(fd=" << mFd << ", size=" << mSize << ", addr=" << mBuffer << "): "
                   << path;
    }

    // Move ctor
    explicit FileMemMapper(FileMemMapper&& other)
        : mFd(other.mFd), mBuffer(other.mBuffer), mSize(other.mSize) {
            other.mFd = -1;
            other.mBuffer = nullptr;
            other.mSize = 0;
        }

    ~FileMemMapper() {
        if (!mBuffer && mFd == -1) {
            return;
        }
        LOG(DEBUG) << "FileMemMapper: Unmapping "
                   << "(fd=" << mFd << ", size=" << mSize << ", addr=" << mBuffer << ")";
        if (mBuffer && munmap(mBuffer, mSize) == -1) {
            LOG(ERROR) << "munmap fail";
            return;
        }
        if (mFd != -1 && close(mFd) == -1) {
            LOG(ERROR) << "close fail";
            return;
        }
    }

    std::pair<char*, size_t> get() const {
        return {reinterpret_cast<char*>(mBuffer), mSize};
    }

    void* getAddr() const {
        return mBuffer;
    }

    size_t getSize() const {
        return mSize;
    }

private:
    int mFd = -1;
    void* mBuffer = nullptr;
    size_t mSize = 0;
};