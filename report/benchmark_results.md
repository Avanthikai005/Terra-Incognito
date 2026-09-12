# Benchmark Results — Terra Incognita

Disaster-type-sequential continual-learning benchmark on xBD-selected.

## Methods × sequences (average forgetting)

| Method | similar_domain | cross_disaster |
|---|---|---|
| baseline | n/a | n/a |
| naive | 0.00% | 0.00% |
| joint | 0.00% | 0.00% |
| cl | 0.00% | 0.00% |
| cl_adaptive | 0.00% | 0.00% |

## Per-sequence matrices

### similar_domain — hurricane-matthew-A, hurricane-matthew-B, hurricane-matthew-C

*3 longitude-band pseudo-regions of Hurricane Matthew (near-identical domains, ~4% feature distance); the previous benchmark as-is.*

#### BASELINE
| after task | hurricane-matthew-A | hurricane-matthew-B | hurricane-matthew-C | avg forgetting |
|---|---|---|---|---|---|---|---|---|---|
| task 1 | 81.2% | 75.0% | 87.5% | n/a |
| task 2 | 25.0% | 18.8% | 0.0% | n/a |
| task 3 | 25.0% | 18.8% | 0.0% | n/a |

#### NAIVE
| after task | hurricane-matthew-A | hurricane-matthew-B | hurricane-matthew-C | avg forgetting |
|---|---|---|---|---|---|---|---|---|---|
| task 1 | 81.2% |   -   |   -   | 0.00% |
| task 2 | 81.2% | 75.0% |   -   | 0.00% |
| task 3 | 81.2% | 75.0% | 87.5% | 0.00% |

#### JOINT
| after task | hurricane-matthew-A | hurricane-matthew-B | hurricane-matthew-C | avg forgetting |
|---|---|---|---|---|---|---|---|---|---|
| task 1 | 81.2% | 75.0% | 87.5% | 0.00% |
| task 2 | 81.2% | 75.0% | 87.5% | 0.00% |
| task 3 | 81.2% | 75.0% | 87.5% | 0.00% |

#### CL
| after task | hurricane-matthew-A | hurricane-matthew-B | hurricane-matthew-C | avg forgetting |
|---|---|---|---|---|---|---|---|---|---|
| task 1 | 81.2% |   -   |   -   | 0.00% |
| task 2 | 81.2% | 75.0% |   -   | 0.00% |
| task 3 | 81.2% | 75.0% | 87.5% | 0.00% |

#### CL_ADAPTIVE
| after task | hurricane-matthew-A | hurricane-matthew-B | hurricane-matthew-C | avg forgetting |
|---|---|---|---|---|---|---|---|---|---|
| task 1 | 81.2% |   -   |   -   | 0.00% |
| task 2 | 81.2% | 75.0% |   -   | 0.00% |
| task 3 | 81.2% | 75.0% | 87.5% | 0.00% |

### cross_disaster — hurricane-matthew, palu-tsunami

*hurricane -> tsunami. Two distinct disaster types; palu RGB is derived from its 12-band Sentinel-2 tiles. Note palu's collapsed label distribution is heavily imbalanced (mostly undamaged).*

#### BASELINE
| after task | hurricane-matthew | palu-tsunami | avg forgetting |
|---|---|---|---|---|---|---|---|
| task 1 | 85.0% | 0.0% | n/a |
| task 2 | 10.0% | 20.0% | n/a |

#### NAIVE
| after task | hurricane-matthew | palu-tsunami | avg forgetting |
|---|---|---|---|---|---|---|---|
| task 1 | 85.0% |   -   | 0.00% |
| task 2 | 85.0% | 0.0% | 0.00% |

#### JOINT
| after task | hurricane-matthew | palu-tsunami | avg forgetting |
|---|---|---|---|---|---|---|---|
| task 1 | 85.0% | 0.0% | 0.00% |
| task 2 | 85.0% | 0.0% | 0.00% |

#### CL
| after task | hurricane-matthew | palu-tsunami | avg forgetting |
|---|---|---|---|---|---|---|---|
| task 1 | 85.0% |   -   | 0.00% |
| task 2 | 85.0% | 0.0% | 0.00% |

#### CL_ADAPTIVE
| after task | hurricane-matthew | palu-tsunami | avg forgetting |
|---|---|---|---|---|---|---|---|
| task 1 | 85.0% |   -   | 0.00% |
| task 2 | 85.0% | 0.0% | 0.00% |

