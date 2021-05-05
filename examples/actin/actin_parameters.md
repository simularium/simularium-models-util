# Global parameters

- timestep - 0.1 ns (largest time step that maintains constraints)
- force constant - 250 (unitless, largest possible at timestep = 0.1 ns)
- temperature - 22C (temperature in Pollard experiments, TODO increase to 37C for physiological conditions, which will require rebalancing of timestep and force constant and update of rates)
- viscosity - 8.1 cP (viscosity in cytoplasm)
- reaction distance - 1 nm (TODO experiment with 0, which is accurate)

# Concentrations and radii

- actin
  - radius - 2 nm
  - concentration - 200 uM (TODO get into realistic range: 20 - 50 uM)
- arp2/3
  - radius - 2 nm (for each monomer in the dimer, so total dimer radius ends up being about 4 nm)
  - concentration - 10 uM (TODO get into realistic range: 0.1 - 3 uM)
- capping protein
  - radius - 3 nm (TODO research structure)
  - concentration - 0 uM (TODO experiment with adding cap)
- seed fibers - actin filaments already polymerized at simulation start, with seed fiber length in nm

# Reaction Rates

Units for all rates = 1 / ns

## Rates from Joh Schoeneberg
- from Pollard et al 1986
- kmacro to kmicro conversion from Frohner and Noé 2018
- assume the barbed-end polymerization rate can be used for dimerization and nucleation

### Barbed End

polymerize rate = 2.1e-4</br>
depolymerize rate = 1.4e-9

### Pointed End

polymerize rate = 2.4e-5</br>
depolymerize rate = 0.8e-9

## Current assumptions

- use rates from Joh for barbed and pointed polymerization/depolymerization
- assume the barbed rates can also be used for dimerize, trimerize, nucleate, and first branch actin (probably a reasonable assumption)
- use Pollard textbook's relative ADP / ATP rates (probably a reasonable assumption)
- use actin rates for Arp2/3 binding and release (starting point, but not a good assumption)
- use actin rates for Cap binding and release (starting point, but not a good assumption)
- use actin hydrolysis rate from Pollard textbook for actin and Arp2/3 (good assumption for actin, but not Arp2/3)
- nucleotide exchange is turned off, rate needs research

## Current rate parameters

Accurate:
- Pointed growth ATP - use Joh's pointed polymerization rate
- Pointed growth ADP - use Joh's pointed polymerization rate, multiplied by relative ADP/ATP rate from Pollard textbook (0.16 / 1.3)
- Pointed shrink ATP - use Joh's pointed depolymerization rate
- Pointed shrink ADP - use Joh's pointed depolymerization rate, multiplied by relative ADP/ATP rate from Pollard textbook (0.3 / 0.8)
- Barbed growth ATP - use Joh's barbed polymerization rate
- Barbed growth ADP - use Joh's barbed polymerization rate, multiplied by relative ADP/ATP rate from Pollard textbook (4. / 12.)
- Barbed shrink ATP - use Joh's barbed depolymerization rate
- Barbed shrink ADP - use Joh's barbed depolymerization rate, multiplied by relative ADP/ATP rate from Pollard textbook (8. / 1.4)
- Hydrolysis actin - use actin hydrolysis rate from Pollard textbook

Probably accurate, but TODO test and/or do more research in literature:
- Dimerize - use Joh's barbed polymerization rate
- Dimerize reverse - use Joh's barbed depolymerization rate
- Trimerize - use Joh's barbed polymerization rate
- Trimerize reverse - use Joh's barbed depolymerization rate
- Nucleate ATP - use Joh's barbed polymerization rate
- Nucleate ADP - use Joh's barbed polymerization rate, multiplied by relative ADP/ATP rate for barbed growth from Pollard textbook (4. / 12.)
- Barbed growth branch ATP - use Joh's barbed polymerization rate
- Barbed growth branch ADP - use Joh's barbed polymerization rate, multiplied by relative ADP/ATP rate from Pollard textbook (4. / 12.)
- Debranching ATP - use Joh's barbed depolymerization rate
- Debranching ADP - use Joh's barbed polymerization rate, multiplied by relative ADP/ATP rate from Pollard textbook (4. / 12.)

Probably inaccurate, TODO more research in literature:
- Arp bind ATP - use Joh's barbed polymerization rate
- Arp bind ADP - use Joh's barbed polymerization rate, multiplied by relative ADP/ATP rate from Pollard textbook (4. / 12.)
- Arp unbind ATP - use Joh's barbed depolymerization rate
- Arp unbind ADP - use Joh's barbed depolymerization rate, multiplied by relative ADP/ATP rate from Pollard textbook (8. / 1.4)
- Cap bind - use Joh's barbed polymerization rate
- Cap unbind - use Joh's barbed depolymerization rate
- Hydrolysis arp - use actin hydrolysis rate from Pollard textbook
- Nucleotide exchange actin - no reference
- Nucleotide exchange arp - no reference