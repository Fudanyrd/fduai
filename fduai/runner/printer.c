//===- fduai/runner/printer.c - Formatting Tensor Data --------------------===//
//
// Provides functions to format tensor data as JSON for printing.
// llvm.func @json_list_start() is used to start a JSON list,
// llvm.func @json_list_end() is used to end a JSON list,
// llvm.func @json_list_sep() is used to separate items in a JSON list,
// llvm.func @json_f32(f32) is used to format a float value as JSON,
//
//===----------------------------------------------------------------------===//

#include <stddef.h>
#include <stdio.h>

void json_list_start() {
    printf("[");
}

void json_list_end() {
    printf("]");
}

void json_list_sep() {
    printf(",");
}

void json_f32(float value) {
    printf("%f", value);
}

void json_list_data(const float *buf, size_t len) {
    json_list_start();

    json_f32(buf[0]);
    for (size_t i = 1; i < len; i++) {
        json_list_sep();
        json_f32(buf[i]);
    }

    json_list_end();
}

void new_line() {
    printf("\n");
    fflush(stdout);
}
