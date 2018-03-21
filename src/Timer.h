#ifndef TIMER_H
#define TIMER_H

#include "Teuchos_TimeMonitor.hpp"

namespace tioga_nalu {

Teuchos::TimeMonitor get_timer(const std::string name);

}

#endif /* TIMER_H */
