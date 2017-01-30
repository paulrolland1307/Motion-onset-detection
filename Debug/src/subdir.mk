################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/MotionDetection.cpp \
../src/helper.cpp \
../src/timing.cpp 

OBJS += \
./src/MotionDetection.o \
./src/helper.o \
./src/timing.o 

CPP_DEPS += \
./src/MotionDetection.d \
./src/helper.d \
./src/timing.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/usr/local/include/opencv2 -I/usr/local/include/opencv -O0 -g3 -Wall -std=c++11 -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


