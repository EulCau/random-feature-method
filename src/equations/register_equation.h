#pragma once

#include "equation_factory.h"

#define REGISTER_EQUATION_CLASS(class_name)									\
	namespace																\
	{																		\
	struct class_name##Registrar											\
	{																		\
		class_name##Registrar()												\
	{																		\
			EquationFactory::instance().register_class(						\
				#class_name,												\
				[](const EqnConfig& config) -> std::shared_ptr<Equation>	\
				{															\
					return std::make_shared<class_name>(config);			\
				});															\
		}																	\
	};																		\
	static class_name##Registrar global_##class_name##_registrar;			\
	}
