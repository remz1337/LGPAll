package nz.co.jedsimson.lgp.core.program.instructions

/**
 * The arity of an operation, specifying how many arguments it can take.
 *
 * @property number An integer that represents how many arguments the arity describes.
 */
interface Arity {
    val number: Int
}

/**
 * A base arity implementation that provides arity for instructions with one or two arguments.
 */
enum class BaseArity(override val number: Int) : Arity {
    /**
     * Represents the arity of an instruction that has one argument.
     */
    Unary(1),

    /**
     * Represents the arity of an instruction that has two arguments.
     */
    Binary(2),

    /**
     * Represents the arity of an instruction that has three arguments.
     */
    Ternary(3);
}

/**
 * Thrown when the [Arity] of an [Operation] does not match the number of [lgp.core.evolution.registers.Argument]s given to it.
 *
 * @param message A message to annotate the exception.
 */
class ArityException(message: String) : Exception(message)