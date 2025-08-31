#pragma once

#include "equation.h"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

class EquationFactory
{
public:
	using Creator = std::function<std::shared_ptr<Equation>(const EqnConfig&)>;

	static EquationFactory& instance()
	{
		static EquationFactory inst;
		return inst;
	}

	void register_class(const std::string& name, Creator creator)
	{
		registry_[name] = std::move(creator);
	}

    [[nodiscard("Return Need to be Used")]]
	std::shared_ptr<Equation> create(const std::string& name, const EqnConfig& config) const
	{
		auto it = registry_.find(name);
		if (it != registry_.end())
		{
			return it->second(config);
		}
		throw std::runtime_error("Unknown Equation subclass: " + name);
	}

private:
	std::unordered_map<std::string, Creator> registry_;
};
