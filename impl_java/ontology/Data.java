package ontology;

import ontology.concepts.Frame;
import ontology.concepts.Role;
import ontology.concepts.Situation;

import java.util.List;

import static ontology.ParsedSentence.createUnknownIndividual;

public class Data {

    public static class Roles {
        // Common roles
        public static final Role location = new Role("role_location", SemTypes.location);
        // Celebration roles
        public static final Role occasion = new Role("role_occasion", SemTypes.occasion);
        public static final Role attendee = new Role("role_attendee", SemTypes.human);
        public static final Role bringer = new Role("role_bringer", SemTypes.human);
        public static final Role brought = new Role("role_brought", SemTypes.object);
        // Murder roles
        public static final Role murderer = new Role("role_murderer", SemTypes.human);
        public static final Role victim = new Role("role_victim", SemTypes.human);
        public static final Role investigator = new Role("role_investigator", SemTypes.human);
        public static final Role suspect = new Role("role_suspect", SemTypes.human);
    }

    public static class Frames {
        // Common frames
        public static final Frame beAt = new Frame(Verbs.beAt, List.of(Roles.attendee, Roles.location));
        // Frames for celebration
        public static final Frame celebrate = new Frame(Verbs.celebrate, List.of(Roles.occasion, Roles.location));
        public static final Frame bring = new Frame(Verbs.bring, List.of(Roles.bringer, Roles.brought, Roles.location, Roles.occasion));
        public static final Frame attend = new Frame(Verbs.attend, List.of(Roles.attendee, Roles.occasion, Roles.location));
        // Frames for murder
        public static final Frame murder = new Frame(Verbs.murder, List.of(Roles.murderer, Roles.victim, Roles.location));
        public static final Frame investigation = new Frame(Verbs.investigate, List.of(
                Roles.investigator,
                Roles.suspect,
                Roles.victim,
                Roles.location
        ));
    }

    public static class Situations {
        public static final Situation celebration = new Situation("situation_celebration", List.of(
                Frames.celebrate
                , Frames.bring
                , Frames.attend
                , Frames.beAt
        ));

        public static final Situation murder = new Situation("situation_murder", List.of(
                Frames.murder
                , Frames.investigation
                , Frames.beAt
        ));
    }

    public static class SemTypes {
        public static final String location = "sType_location";
        public static final String occasion = "sType_occasion";
        public static final String human = "sType_human";
        public static final String object = "sType_object";
    }

    public static class Individuals {
        // Humans
        public static final ParsedSentence.TypedIndividual alice = createInd("alice", SemTypes.human);
        public static final ParsedSentence.TypedIndividual bob = createInd("bob", SemTypes.human);
        public static final ParsedSentence.TypedIndividual charlie = createInd("charlie", SemTypes.human);
        public static final ParsedSentence.TypedIndividual david = createInd("david", SemTypes.human);
        public static final ParsedSentence.TypedIndividual john = createInd("john", SemTypes.human);
        // Objects
        public static final ParsedSentence.TypedIndividual cake = createInd("cake", SemTypes.object);
        // Locations
        public static final ParsedSentence.TypedIndividual home = createInd("home", SemTypes.location);
        public static final ParsedSentence.TypedIndividual park = createInd("park", SemTypes.location);
        // Occasions
        public static final ParsedSentence.TypedIndividual birthday = createInd("birthday", SemTypes.occasion);
        public static final ParsedSentence.TypedIndividual newYearsEve = createInd("newYearsEve", SemTypes.occasion);

        public static final ParsedSentence.TypedIndividual unknownHuman = createUnknownIndividual(SemTypes.human);
        public static final ParsedSentence.TypedIndividual unknownObject = createUnknownIndividual(SemTypes.object);
        public static final ParsedSentence.TypedIndividual unknownLocation = createUnknownIndividual(SemTypes.location);
        public static final ParsedSentence.TypedIndividual unknownOccasion = createUnknownIndividual(SemTypes.occasion);

        private static ParsedSentence.TypedIndividual createInd(String name, String semType) {
            return new ParsedSentence.TypedIndividual(name, semType);
        }


    }

    public static class Verbs {
        public static final String beAt = "frame_be_at";
        public static final String celebrate = "frame_celebrate";
        public static final String bring = "frame_bring";
        public static final String attend = "frame_attend";
        public static final String murder = "frame_murder";
        public static final String investigate = "frame_investigate";
    }

}
