/*
 * Copyright 2002-2020 Intel Corporation.
 * 
 * This software is provided to you as Sample Source Code as defined in the accompanying
 * End User License Agreement for the Intel(R) Software Development Products ("Agreement")
 * section 1.L.
 * 
 * This software and the related documents are provided as is, with no express or implied
 * warranties, other than those that are expressly stated in the License.
 */

/*
 * this application calls a user-written function that contains a bad code pattern.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void fall_thru();

int main( int argc, char * argv[] )
{
    char * buffer;

    buffer = (char *)malloc( 64 );
    strcpy( buffer, "abc" );
    printf("%s\n", buffer );
    fall_thru();
    printf("returned from fall_thru.\n");
    free( buffer );
    return 0;
}


    
    