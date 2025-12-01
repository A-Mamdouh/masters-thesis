package modelGeneration.ontModel.invokations;

import java.util.Optional;

import fol.Language;
import fol.formula.*;
import fol.term.Constant;
import fol.term.Variable;
import ontology.ParsedSentence.TypedIndividual;
import ontology.concepts.Role;

public class InvokedRole {
    private final Role role;
    private final String name;
    private final boolean isNegated;

    public InvokedRole(Role role, String name, boolean isNegated) {
        this.role = role;
        this.name = name;
        this.isNegated = isNegated;
    }

    public Role getRole() {
        return role;
    }

    public String getName() {
        return name;
    }

    public boolean isNegated() {
        return isNegated;
    }

    public static Optional<InvokedRole> invoke(Role role, TypedIndividual individual, boolean isNegated) {
        if(!individual.semType().equals(role.semType())) {
            return Optional.empty();
        }
        return Optional.of(new InvokedRole(role, individual.name(), isNegated));
    }

    public Formula getBaseFormula(Variable frameVar) {
        // If the role is known as 'C', fill it in and return role(fVar, roleName, adopter)
        if(new TypedIndividual(name, role.semType()).isKnown()) {
            return Language.Predicates.role(frameVar, new Constant(role.name()), new Constant(name));
        }
        // Otherwise return \exists_r:semType(r, semType).role(fVar, r, roleN)
        final var adopterVar = Variable.make();
        final var innerForm = Language.Predicates.role(frameVar, new Constant(role.name()), adopterVar);
        final var precondition = Language.Predicates.semType(adopterVar, new Constant(role.semType()));
        return new Exists(adopterVar, precondition, innerForm);
    }

    public Formula getNegatedFormula(Variable frameVar) {
        // If the role is known as 'C', fill it in and return \not role(fVar, roleName, adopter) [did not happen or someone else]
        if(new TypedIndividual(name, role.semType()).isKnown()) {
            final var adopter = new Constant(name);
            final var roleConst = new Constant(role.name());
            final var adopterVar = Variable.make();
            final var innerForm = new And(Language.Predicates.role(frameVar, roleConst, adopterVar), new Not(new Equals(adopterVar, adopter)));
            final var precondition = Language.Predicates.semType(adopterVar, adopter);
            final Formula notAdopter = new Not(Language.Predicates.role(frameVar, new Constant(role.name()), new Constant(name)));
            final Formula otherThanAdopter = new Exists(adopterVar, precondition, innerForm);
            return new Not(new And(new Not(notAdopter), new Not(otherThanAdopter)));
        }
        // Otherwise return \not \exists_r:semType(r, semType).role(fVar, r, roleN) [not something == nothing]
        final var adopterVar = Variable.make();
        final var innerForm = Language.Predicates.role(frameVar, new Constant(role.name()), adopterVar);
        final var precondition = Language.Predicates.semType(adopterVar, new Constant(role.semType()));
        return new Not(new Exists(adopterVar, precondition, innerForm));
    }

    public Formula getFormula(Variable frameVar) {
        if(isNegated) {
            return getNegatedFormula(frameVar);
        }
        return getBaseFormula(frameVar);
    }
}
