package modelGeneration.axioms;

import modelGeneration.folModel.Model;

public interface ConsistencyCheck {
    boolean check(Model model);
    ConsistencyCheck getCopy();
}
