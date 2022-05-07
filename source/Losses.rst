2022-01
==========

*_sdk 代码阅读
***************************

PImpl
~~~~~~~~

- "Pointer to implementation" or "pImpl" is a C++ programming technique 
  that removes implementation details of a class from its object representation 
  by placing them in a separate class, accessed through an opaque pointer:

.. code-block::

   // ---------------------------
   // interface (widget.h)
   class widget {
       // public members
   private:
       struct impl;  // forward declaration of the implementation class
       std::experimental::propagate_const<std::unique_ptr<impl> > pImpl;
   };

   // ---------------------------
   // implementation (widget.cpp)
   struct widget::impl
   {
       // implementation details
   };


- This technique is used to construct C++ library interfaces with stable ABI and to reduce compile-time dependencies.

- The Pimpl idiom technique is also referred to as an opaque pointer, handle classes, compiler firewall idiom,
  d-pointer, or cheshire cat.
  This idiom is useful because it can minimize coupling, and separates the interface from the implementation.
  It is a way to hide the implementation details of an interface from the clients.
  It is also important for providing **binary code compatibility** with different version of a shared library.
  The Pimpl idiom simplifies the interface that is created since the details can be hidden in another file.

**What are the benefits of Pimpl?**

Generally, whenever a header file changes, any file that includes that file will need to be recompiled.
This is true even if those changes only apply to private members of the class that, by design,
the users of the class cannot access.
This is because of the C++ build model and because C++ assumes that callers know two main things about a class (and private members).

1. Size and layout: The code that is calling the class must be told the size and layout of the class (including private data members).
   This constraint of seeing the implementation means the callers and callees are more tightly coupled,
   but is very importance to the C++ object model because having direct access to object by default helps C++ achieve heavily-optimized efficiency.

2. Functions: The code that is calling the class must be able to resolve calls to member functions of the class.
   This includes private functions that are generally inaccessible and overload non-private functions.
   If a private function is a better match, the code will fail to compile.

With the Pimpl idiom, you remove the compilation dependencies on internal (private) class implementations.
The big advantage is that it breaks compile-time dependencies.
This means the system builds faster because Pimpl can eliminate extra includes.
Also, it localized the build impact of code changes because the implementation (parts in the Pimpl)
can be changed without recompiling the client code.

**Implementation**

As the object of the interface type controls the lifetime of the object of the implementation type,
the pointer to implementation is usually :code:`std::unique_ptr`.

Because :code:`std::unique_ptr` requires that the pointed-to type is a complete type in any context where the deleter is instantiated,
the special member functions must be user-declared and defined out-of-line,
in the implementation file, where the implementation class is complete.

Because when const member function calls a function through a non-const member pointer,
the non-const overload of the implementation function is called, the pointer has to be wrapped in
:code:`std::experimental::propagate_const` or equivalent.

All private data members and all private non-virtual member functions are placed in the implementation class.
All public, protected, and virtual members remains in the interface class.

If any of the private members needs to access a public or protected member,
a reference or pointer to the interface may be passed to the private function as a parameter.
Alternatively, the back-reference may be maintained as part of the implementation class.

If non-default allocators are intended to be supported for the allocation of the implementation object,
any of the usual allocator awareness patterns may be utilized, including allocator template parameter defaulting to
:code:`std::allocator` and  constructor argument of type :code:`std::pmr::memory_resource*`.

- pimpl 是 c++ 常用的设计模式。当我们开发一个SDK，或者设计某个模块，需要暴露一个 \*.h 头文件，
  但是该文件中有一些 private 函数和字段，这些函数和字段的本意是不想被用户知道的，
  因为可能里面有些隐私内容，用户有可能通过这些 private 方法和字段就能猜出我们的架构和实现。

**Summary:**

The Pimpl idiom is a great way to minimize coupling and break compile-time dependencies,
which leads to faster build times.


unique_ptr
~~~~~~~~~~~

**Unique pointer** manages the storage of a pointer,
providing a limited *garbage-collection* facility,
with little to no overhead over built-in pointers (depending on the deleter used).

These objects have the ability of *taking ownership* of a pointer:
once they take owership they manage the pointed object by becoming responsible for its deletion at some point.

:code:`std::unique_ptr` is a smart pointer that owns and manages another object through a pointer
and disposes of that object when the :code:`unique_ptr` goes out of scope.

:code:`unique_ptr` is one of the smart pointer implementatin provided by c++11 to prevent memeory leaks.
A unique_ptr object wraps around a raw pointer and its responsible for its lifetime.
When this object is destructed then in its destructor it deletes the associated raw pointer.

:code:`unique_ptr` has its :code:`->` and :code:`*` operator overloaded,
so it can be used similar to normal pointer.

A :code:`unique_ptr` object is always the unique owner of associated raw pointer.
We can not copy a :code:`unique_ptr` object, it's only movable.

A :code:`unique_ptr` may alternvatively own no object, in which case it is called *empty*.

There are two ways to check if a :code:`unique_ptr<>` object is empty or it has a raw pointer associated with it i.e.

**Method 1**

.. code::

    // Check if unique pointer object is empty
    if (!ptr)
        std::cout << "ptr is empty" << std::endl;

**Method 2**

.. code::

    // Check if unique pointer object is empty
    if (ptr == nullptr)
        std::cout << "ptr is empty" << std::endl;

To create a :code:`unique_ptr<>` object that is non empty, we need to pass the raw pointer in its constructor
while creating the object i.e.

.. code::

    // Create a unique_ptr object through raw pointer
    std::unique_ptr<T> tPtr(new T());    //? are there parenthesis or not?

After C++14 use

.. code::

    std::unique_ptr<T> tPtr(std::make_unique<T>())

We can not create a :code:`unique_ptr<>` object through assignment, otherwise it will cause compile error

.. code::

   // std::unique_ptr<T> tPtr = new T(); // Compile Error

As :code:`unique_ptr<>` is not copyable, only movable,
hence we can not create copy of a unique_ptr object either through copy constructor or assignment operator.

.. code::

    // Create a unique_ptr object through raw pointer
    std::unique_ptr<T> tPtr2(new T());

    ttd::unique_ptr<F> tPtr3 = tPtr2; // Compile error
   
Both copy constructor and assignment operator are deleted in :code:`unique_ptr<>` class.
Reference `unique_ptr<> Tutorial and Examples <https://thispointer.com/c11-unique_ptr-tutorial-and-examples/>`_


make_unique
~~~~~~~~~~~~~

We can not create a :code:`unique_ptr<>` object through assignment, otherwise it will cause compile error

virtual function in c++
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A virtual function is a member function which is declared within a base class and is re-defined (overridden) by a derived class.
When you refer to a derived class object using a pointer or a reference to the base class,
you can call a virtual function for that object and execute the derived class's version of the function.

- Virtual functions ensure that the correct function is called for an object,
  regardless of the type of reference (or pointer) used for function call.
- They are mainly used to achieve Runtime polymorphism
- Functions are declared with a virtual keyword in base class.
- The resolving of function call is done at runtime.

**Rules for Virtual Functions**

1. Virtual functions cannot be static.
2. A Virtual function can be a friend function of another class.
3. Virtual functions should be accessed using pointer or reference of base class type to achieve runtime polymorphism.
4. The prototype of virtual functions should be the same in the base as well as derived class.
5. They are always defined in the base class and overridden in a derived class.
   It is not mandatory for the derived class to override (or re-define the virtual function),
   in that case, the base class version of the function is used.
6. A class may have virtual destructor but it cannot have a virtual constructor.


Runtime polymorphism
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Runtime polymorphism is achieved by Function Overriding.

- Function overriding occurs when a derived class has a definition for one of the member functions of the base class.
  That base function is said to be overridden.


.. code-block::

   #include <bits/stdc++.h>
   using namespace std;

   class base {
   public:
        virtual void point() {
                cout << "Print base class" << endl;
        }
        void show () {
                cout << "Show base class" << endl;
        }
   };

   class derived: public base {
   public:
        void print () {
                cout << "Print derived class" << endl;
        }
        void show () {
                cout << "Print derived class" << endl;
        }
   };

   int main() {
        base *bptr;
        derived d;
        bptr = &d;

        bptr->print();
        bptr->show();
        return 0;
   }


.. code-block::

        print derived class
        show base class


**Explanation**: Runtime polymorphism is achieved only through a pointer (or reference) of base class type.
Also, a base class pointer can point to the objects of base class as well as to the objects of derived class.
In above code, base class pointer 'bptr' contains the address of object 'd' of derived class.
Late binding (Runtime) is done in accordance with the content of pointer (i.e. location pointed to by pointer) and
Early binding (Compile time) is done according to the type of pointer,
since :code:`point()` function is declared with virtual keyword so it will be bound at runtime
(output is *print derived class* as pointer is pointing to object of derived class) and
:code:`show()` is  non-virtual so it will be bound during compile time
(output is *show base class* as pointer is of base type).


.. note::

        If we have created a virtual function in the base class and it is being overridden in the derived class
        then we don't need virtual keyword in the derived class,
        functions are automatically considered as virtual functions in the derived class.
        
If a class contains a virtual function then compiler itself does two things.

1. If object of that class is created then a *virtual pointer (VPTR)* is inserted as a data member of the class to point to VTABLE of that class.
   For each new object created, a new virtual pointer is inserted as a data member of that class.
2. Irrespective of object is created or not, class contains as a member *a static array of function pointers calld VTABLE*.
   Cells of this table store the address of each virtual function contained in that class.


.. image::    VirtualFunctionInC.png

Virtual Destructor
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deleting a derived class object using a pointer of base class type that has a non-virtual destructor results in undefined behavior.
To correct this situation, the base class should be defined with a virtual destructor.
Making base class destructor virtual guarantees that the object of derived class is destructed properly, i.e.,
both base class and derived class destructors are called.

As a guideline, any time you have a virtual function in a class,
you should immediately add a virtual destructor (even if it does nothing).
This way, you ensure against any surprises later.

Explicit Keyword in C++
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Explicit Keyword in c++** is used to make constructors to not implicitly convert types in C++.
It is optional for constructors that take exactly one argument and works on constructors (with single argument)
since those are the only constructors that can be used in type casting.

The compiler is allowed to make one implicit conversion to resolve the parameters to a function.
What this means is that the compiler can use constructors callable with a **single parameter** to convert from one type to another
in order to get the right type for a parameter.

Prefixing the :code:`explicit` keyword to the constructor prevents the compiler form using that constructor for implicit conversions.

The reason you might want to do this is to avoid accidental construction that can hide bugs.

Contrived example:

- You have a :code:`Mystring` class with a constructor that constructs a string of the given size.
  You have a function :code:`print(const MyString&)` (as well as an overload :code:`print (char *string)`),
  and you call :code:`print(3)` (when you *actually* intended to call :code:`print("3")`).
  You expect it to print "3", but it prints an empty string of length 3 instead.

在 C++ 中，explicit 关键字用来修饰类的构造函数，被修饰的构造函数的类，不能发生相应的隐式型转换，只能以显示的方式进行转换。

- explicit 关键字只能用于类内部的构造函数声明上。


Const member functions in C++
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A free function cannot be marked with "const" (after a function declaration?), only a method can.

A method (a function that belongs to a class) marked with "const", has the type of its "this" pointer turned into const.

.. code::

        struct A {
                int n;
                void show() const {
                        std::cout << this->n << std::endl;
                }
        }

In this example, the type of "this" in the method "show" is "const A*".
The type was modified to "const" because the method is marked with const.

Consequences of that?

- The method explicated that it cannot modify the value of any attribute of the object,
  expect if it is marked as "mutable" (with the keyword :code:`mutable`).
- Inside this method, no "non-const" method of same class can be invoked, only const-ones.
- If the method wants to return a reference to an attribute of the class, the reference has to be const as well.

A "const function", denoted with the keyword :code:`const` after a function declaration,
makes it a compiler error for this class function to change a member variable of the class.
However, reading of a class variables is okay inside of the function,
but writing inside of this function will generate a compiler error.

Another way of thinking about such "const function" is by viewing a class function as a normal function taking an implicit :code:`this` pointer.
So a method :code:`int Foo::Bar (int random_arg)` (without the const at the end)
results in a function like :code:`int Foo_Bar (Foo* this, int random_arg)`,
and a call such as :code:`Foo f; f.Bar(4)` will internally correspond to something like :code:`Foo f; Foo_Bar(&f, 4)`.
Not adding the const at the end (:code:`int Foo::Bar (int random_arg) const`) can then be understood as a declaration
with a const this pointer: :code:`int Foo_Bar (const Foo* this, int random_arg)`.
Since the type of :code:`this` in such case is const, no modifications of member variables are possible.

When a function is declared as :code:`const`, it can be called on any type of object.
Non-const functions can only be called by non-const objects.

inline keyword in c++
~~~~~~~~~~~~~~~~~~~~~~~~~~~
When the program executes the function call instruction the CPU stores the memory address of the instruction following the function call,
copies the arguments of the function on the stack and finally transfers control to the specified function.
The CPU then executes the function code, stores the function return value in a predefined memory location/register
and returns control to the calling function.
This can become overhead if the execution time of function is less than the switching time from the called function to called function (callee).
For functions that are large and/or perform complex tasks,
the overhead of the function call is usually insignificant compared to the amount of time the function takes to run.
However, for small, commonly-used functions, the time needed to make the function call is often a lot more
than the time needed to actually execute the function's code.
This overhead occurs for small functions because execution time of small function is less than the switching time.

C++ provides an inline functions to reduce the function call overhead.
Inline function is a function that is expanded in line when it is called.
When the inline function is called whole code of the inline function gets inserted or substituted at the point of inline function call.
This substitution is performed by the C++ compiler at compile time.
Inline function may increase efficiency if it is small.


Inline function is one of the important feature in C++.

When the program executes the function call instruction
the CPU stores the memory address of the instruction following the function call,
copies the arguments of the function on the stack and
finally transfers control to the specified function.
The CPU then executes the function code,
stores the function return value in a predefined memory location/register
and returns control to the calling function.
This can become overhead if the execution time of function is less than the switching time
from the caller function to called function (callee).
For functions that are large and/or perform complex tasks,
the overhead of the function call is usually insignificant compared to the amount of time the function takes to run.
However, for small, commonly-used functions,
the time needed to make the function call is often a lot more than the time needed to make the function call is often a lot more than
the time needed to actually execute the function's code.
This overhead occurs for small functions because execution time of small function is less than the switching time.


protected keyword in c++
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The class member declared as Protected are inaccessible outside the class
but they can be accessed by any subclass (derived class) of that class. i.e.,
Protected members are accessible in the class that defines them and in classes that inherit from that class.

Private members are only accessible within the class defining them.

Both private and protected are also accessible by friends of their class, and in the case of protected members,
by friends of their derived classes.

Use whatever makes sense in the context of your problem.
You should try to make members private whenever you can to reduce coupling and protect the implementation of the base class,
but if that's not possible then use protected members.


Polymorphism in c++
~~~~~~~~~~~~~~~~~~~~~~~~~~~

2. Sphinx tutorial

3. 自动化的数据处理

- 可视化所有结果，便于debug
- 整理数据增广文档 `M1 detect module 数据整理 <https://moqi.quip.com/C3BJA7wO9ELU>`_
1. tech share `E2/M1 项目相关检测模型 <https://moqi.quip.com/XNOLASQSr946>`_
2. 训练新模型 (更新见文档 `[2021-10] M1 detect module 模型和数据整理 <https://moqi.quip.com/sal3APTb3hgb>`_ )
3. 数据分析 (更新见文档 `[2021-10] M1 检测模型分析文档 <https://moqi.quip.com/sal3APTb3hgb>`_ )


名词解释
***************************

1. IR: infrared radiation
2. ICB: international conference on biometrics
3. PA: presentation attacks

Evaluation Methodologies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. FAR: false acceptance rate (false recognized as true/# false)
2. FRR: false rejection rate （true recognized as false/# true)
3. HTER: half total error rate ( :math:`0.5 * (FRR + FAR)`)
4. EER: equal error rate.
   EER is a biometric security system algorithm used to predetermines the threshold values for its false acceptance rate and its false rejection rate.
   When the rates are equal, the common value is referred to as the equal error rate.
   In general, the lower the equal error rate value, the higher the accuracy of the biometric system.
5. ROC: receiver operating characteristic
6. DET: detection error tradeoff
7. F-ratio
8. d (to decidability or decision-making power)


Technical Terminologies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. In-database processing, sometimes referred to as in-database analytics, refers to the integration of data analytics into data warehousing functionality.
2. ABI: in computer software, an application binary interface (ABI) is an interface between two binary program modules.
   Often, one of these modules is a library or operating system fcility, and the other is a program that is being run by a user.
