# GLOBALS
    # constants
        POPULATION_SIZE = 10 
        MAX_FOOD = 100
        MAX_GROUNDSKEEPERS = 3
        XP_LEECH_RATE = 1  # XP stolen per second
        PUNISHMENT_COOLDOWN = 3.0  # Seconds between punishments

# ENTITIES

## AGENT
    Controlled by neural network

    ## Size
        Initial size = 1
        Size automatically updates based on level
        Dictates color
        Determined by current level equal to height and width  
        ie level 1 = 1x1, level 2 = 2x2, level 3 = 3x3, etc.
    
    ## Level
        Initial level = 0
        Initial total xp = 0
        Level automatically calculated from total XP
        Each time we eat food we gain 1xp
        Each time we eat another agent we gain their total xp
        Each time we breed we gain offspring's starting xp
        Each time we die we lose all our xp
        XP required for level N is (N * 100)
        Levels update automatically based on total XP
     
    ## Color
        Color updates automatically every frame:
        Red component = level progress (0-255)
        Green component = XP progress to next level (0-255)
        Blue component = size (0-255)
        Alpha = 255 (fully opaque)

## GROUNDSKEEPER
    Controlled by neural network
    Patrols environment seeking agents
    Cannot eat other groundskeepers
    Cannot be eaten by agents
    
    ## Abilities
        Leech XP from agents on contact
        Punish agents by reducing their level
        Faster movement than regular agents
    
    ## Mechanics
        Steals XP at constant rate during collision
        Can only punish same agent after cooldown
        Appears as distinct color (red) in vision
    
# VISION ENCODING
    0 = Empty space
    1 = Self
    2 = Food
    3 = Offspring
    4 = Other agent
    5 = Same-size agent
    6 = Other agent (different size)
    7 = Groundskeeper
    
# ADDITIONAL INPUTS
    time_alive: How long the agent has survived
    punishment_cooldown: Time until agent can be punished again
    xp_stolen: Amount of XP stolen by groundskeepers
    relative_size: Size compared to nearby agents

# PERFORMANCE METRICS
    Average lifespan per generation
    Average offspring per agent
    Average level reached
    Population diversity
    Species distribution
    XP stolen by groundskeepers
    Punishment frequency
    Agent survival rate

# EVOLUTION
    Agents evolve when:
    - Level changes
    - Significant XP milestone reached
    - Generation ends
    
    Groundskeepers evolve when:
    - XP stolen reaches threshold
    - Generation ends
    - Punishment count reaches threshold

# BALANCE CONSIDERATIONS
    Groundkeeper speed vs Agent speed
    XP leech rate vs XP gain rate
    Punishment cooldown duration
    Number of groundskeepers vs Population size
    Punishment severity vs Level requirement

# DATA STRUCTURE DIAGRAM
    GameState
    ├─ agents[POPULATION_SIZE] : Agent
    │   ├─ position : Vector2
    │   ├─ rect : Rectangle
    │   ├─ size : unsigned int
    │   ├─ level : int
    │   ├─ total_xp : int
    │   ├─ time_alive : float
    │   ├─ agent_id : int
    │   ├─ parent_id : int
    │   ├─ num_offsprings : int
    │   ├─ num_eaten : int
    │   ├─ is_breeding : bool
    │   ├─ breeding_timer : float
    │   ├─ color : Color
    │   ├─ brain : NEAT_t*  ──┐
    │   │                     └─ Neural network controlling agent actions
    │   ├─ memory : Memory    ── Stores experiences for reinforcement learning
    │   └─ input_size : size_t
    │
    ├─ food[MAX_FOOD] : Food
    │   ├─ position : Vector2
    │   └─ rect : Rectangle
    │
    ├─ last_actions[POPULATION_SIZE] : Action  ── Last action executed per agent
    ├─ over : bool            ── Whether simulation is over
    ├─ paused : bool          ── Whether simulation is paused
    ├─ evolution_timer : float ── Tracks time for evolution events
    ├─ current_generation : unsigned int
    ├─ vision_inputs : long double* ── Input vector for NN per agent
    ├─ next_agent_id : int
    └─ num_active_players : unsigned int

    Agent
    ├─ position : Vector2
    ├─ rect : Rectangle
    ├─ size : unsigned int
    ├─ level : int
    ├─ total_xp : int
    ├─ time_alive : float
    ├─ agent_id : int
    ├─ parent_id : int
    ├─ num_offsprings : int
    ├─ num_eaten : int
    ├─ is_breeding : bool
    ├─ breeding_timer : float
    ├─ color : Color
    ├─ brain : NEAT_t* ── Neural network controlling movement/actions
    ├─ memory : Memory ── Stores past inputs, actions, rewards
    └─ input_size : size_t

    Food
    ├─ position : Vector2
    └─ rect : Rectangle

