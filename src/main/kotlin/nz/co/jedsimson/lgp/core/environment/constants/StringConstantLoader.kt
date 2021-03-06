package nz.co.jedsimson.lgp.core.environment.constants

import nz.co.jedsimson.lgp.core.modules.ModuleInformation

/**
 * An implementation of [ConstantLoader] that loads constants as a collection of [String]s.
 *
 * We get around having to implement a lot of this ourselves by subclassing [GenericConstantLoader]
 * with a parsing function suitable for converting the raw strings to doubles.
 *
 * @param constants A collection of raw constants.
 * @see [GenericConstantLoader]
 */
class StringConstantLoader constructor(constants: List<String>)
    : GenericConstantLoader<String>(constants, String::toString) {

    // Give this loader a custom description since it is provided as part of the core package.
    override val information = ModuleInformation (
        description = "A loader than can parse specified constants into strings."
    )
}
