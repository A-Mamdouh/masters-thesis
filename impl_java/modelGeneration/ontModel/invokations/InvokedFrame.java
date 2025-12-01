package modelGeneration.ontModel.invokations;

import fol.Language;
import fol.formula.And;
import fol.formula.Exists;
import fol.formula.Formula;
import fol.formula.Predicate;
import fol.term.Constant;
import fol.term.Variable;
import java.util.*;
import ontology.ParsedSentence;
import ontology.concepts.Frame;
import ontology.concepts.Role;

public class InvokedFrame {
    private final Frame frame;
    private final Set<InvokedRole> invokedRoles;

    public InvokedFrame(Frame frame, Set<InvokedRole> invokedRoles) {
        this.frame = frame;
        this.invokedRoles = invokedRoles;
    }

    public static Set<InvokedFrame> invokeNegated(Frame frame, ParsedSentence sentence) {
        final Set<InvokedFrame> invokedFrames = new HashSet<>();
        // Try to invoke the frame here. Return empty if the sentence does not match the frame
        if(!sentence.verb().equals(frame.name())) {
            return Set.of();
        }

        final Set<InvokedRole> invokedRoles = new HashSet<>();
        final var remainingIndividuals = new ArrayList<>(sentence.individuals());
        final Set<Role> unfilledRoles = new HashSet<>(frame.roles());
        // Match roles of the frames with individuals in the sentence using the semantic type and
        // order of appearance in the sentence
        for(final var role : frame.roles()) {
            for (int i = 0; i < remainingIndividuals.size(); ++i) {
                final var ind = remainingIndividuals.get(i);
                final var maybeInvokedRole = InvokedRole.invoke(role, ind, false);
                if (maybeInvokedRole.isPresent()) {
                    remainingIndividuals.remove(i);
                    invokedRoles.add(maybeInvokedRole.get());
                    unfilledRoles.remove(role);
                    break;
                }
            }
        }

        final Set<InvokedRole> invokedUnfilledRoles = new HashSet<>();
        for(final var role : unfilledRoles) {
            final var ind = ParsedSentence.createUnknownIndividual(role.semType());
            invokedUnfilledRoles.add(new InvokedRole(role, ind.name(), false));
        }

        // Generate n invoked Frames where each of the n invoked Roles is negated
        for(final var invokedRole : invokedRoles) {
            final Set<InvokedRole> frameRoles = new HashSet<>(invokedUnfilledRoles);
            for(final var otherRole : invokedRoles) {
                frameRoles.add(new InvokedRole(otherRole.getRole(), otherRole.getName(), otherRole == invokedRole));
            }
            // invoke unfilled roles
            invokedFrames.add(new InvokedFrame(frame, frameRoles));
        }

        return invokedFrames;
    }

    public static Optional<InvokedFrame> invoke(Frame frame, ParsedSentence sentence) {
        // Try to invoke the frame here. Return empty if the sentence does not match the frame
        if(!sentence.verb().equals(frame.name())) {
            return Optional.empty();
        }

        final Set<InvokedRole> invokedRoles = new HashSet<>();
        final var remainingIndividuals = new ArrayList<>(sentence.individuals());
        final Set<Role> unfilledRoles = new HashSet<>(frame.roles());

        for(final var role : frame.roles()) {
            for(int i=0; i<remainingIndividuals.size(); ++i) {
                final var ind = remainingIndividuals.get(i);
                final var maybeInvokedRole = InvokedRole.invoke(role, ind, false);
                if(maybeInvokedRole.isPresent()) {
                    remainingIndividuals.remove(i);
                    invokedRoles.add(maybeInvokedRole.get());
                    unfilledRoles.remove(role);
                    break;
                }
            }
        }

        // invoke unfilled roles
        for(final var role : unfilledRoles) {
            final var ind = ParsedSentence.createUnknownIndividual(role.semType());
            invokedRoles.add(new InvokedRole(role, ind.name(), false));
        }

        return Optional.of(new InvokedFrame(frame, invokedRoles));
    }

    public Formula getFormula(Variable situationVar) {
        // \exists_f:frame(sVar, f, name).\bigWedge[roles]
        final var frameVar = Variable.make();
        Formula rolesConj = Predicate.TRUE;
        for(final var invokedRole : invokedRoles) {
            rolesConj = new And(rolesConj, invokedRole.getFormula(frameVar));
        }
        final var precondition = Language.Predicates.frame(situationVar, new Constant(frame.name()), frameVar);
        return new Exists(frameVar, precondition, rolesConj);
    }

    public Set<InvokedRole> getInvokedRoles() {
        return invokedRoles;
    }
}