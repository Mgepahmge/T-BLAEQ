#pragma once

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wparentheses"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#pragma GCC diagnostic ignored "-Wdangling-reference"
#endif

#define PICOJSON_USE_INT64
#include "src/picojson.h"
