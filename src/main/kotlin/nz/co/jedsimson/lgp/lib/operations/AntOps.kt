package nz.co.jedsimson.lgp.lib.operations

import nz.co.jedsimson.lgp.core.program.registers.Arguments
import nz.co.jedsimson.lgp.core.modules.ModuleInformation
import nz.co.jedsimson.lgp.core.program.instructions.TernaryOperation
import nz.co.jedsimson.lgp.core.program.registers.RegisterIndex

abstract class AntOperation<TData>(func: (Arguments<TData>) -> TData) : TernaryOperation<TData>(func)

/**
 * Performs a branch by comparing two arguments using the greater than operator.
 *
 * Instructions using this operation essentially achieve the following:
 *
 * ```
 * if (r[1] > r[2]) {
 *     return 1.0;
 * } else {
 *     return 0.0;
 * }
 * ```
 *
 * This can be used by an interpreter to determine if a branch should be taken or not
 * (by treating the return value as a boolean).
 * ```
 */
class IfFoodAhead : BranchOperation<String>(
        func = { args: Arguments<String> ->
            //if (args.get(0) > args.get(1)) 1.0 else 0.0
            //if food ahead, return arg0, else arg1
            print(args.get(2).toString())
            "ok"
        }
) {
    override val representation = " fd "

    override val information = ModuleInformation(
            description = ""
    )

    override fun toString(operands: List<RegisterIndex>, destination: RegisterIndex): String {
        return "if(r[${ operands[0] }] > r[${ operands[1] }])"
    }
}

