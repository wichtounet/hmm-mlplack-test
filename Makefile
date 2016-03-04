default: release

.PHONY: default release debug all clean

include make-utils/flags.mk
include make-utils/cpp-utils.mk

CXX_FLAGS += -pedantic

CXX_FLAGS += -I/usr/include/armadillo_bits/
LD_FLAGS  += -lmlpack -larmadillo -lboost_serialization -lboost_program_options

$(eval $(call auto_folder_compile,src))
$(eval $(call auto_add_executable,hmm))

release: release_hmm
release_debug: release_debug_hmm
debug: debug_hmm

all: release release_debug debug

run: release_debug
	./release_debug/bin/hmm

gdb_run: release_debug
	gdb -ex run --args ./release_debug/bin/hmm

clean: base_clean

include make-utils/cpp-utils-finalize.mk
